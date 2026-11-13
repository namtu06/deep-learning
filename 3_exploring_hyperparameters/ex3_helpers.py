"""Helpers for Exercise 3: the dataset, the plots, and the code behind the tests.

You do not need to read or change this file to do the exercise. Keep it next to the notebook.
At grading, the original version of this file is used.
"""
import copy
import csv
import os
import shutil
import tempfile
import time
from contextlib import contextmanager

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------------------------------------------------
# The dataset and its features

ESC50_URL = 'https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip'
FEATURES_FILE = 'esc50_logmel.npz'
FRAMES_PER_STEP = 4    # neighbouring frames that are averaged into one time step


def get_esc50(dataset_path, download=True):
    """Download and unpack ESC-50 into dataset_path if it is not there yet.
    Returns the list of clips (dicts with filename, fold, target and category) and the dataset folder."""
    folder = os.path.join(dataset_path, 'ESC-50-master')
    meta = os.path.join(folder, 'meta', 'esc50.csv')
    if not download:
        return [], folder
    if not os.path.exists(meta):
        import urllib.request
        import zipfile
        os.makedirs(dataset_path, exist_ok=True)
        zip_path = os.path.join(dataset_path, 'ESC-50-master.zip')
        if not os.path.exists(zip_path):
            print("Downloading ESC-50 (about 650 MB), this can take a few minutes ...")
            urllib.request.urlretrieve(ESC50_URL, zip_path + '.part')
            os.replace(zip_path + '.part', zip_path)
        print("Unpacking ...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(dataset_path)
        print(f"Done. You can now delete {zip_path} to free 650 MB.")
    with open(meta) as file:
        clips = [dict(row, fold=int(row['fold']), target=int(row['target'])) for row in csv.DictReader(file)]
    print(f"{len(clips)} clips found in {folder}")
    return clips, folder


def load_clip(folder, clip):
    return librosa.load(os.path.join(folder, 'audio', clip['filename']), sr=22050)


def clip_features(folder, clip, log_mel):
    """The log-mel spectrogram of a clip, with every FRAMES_PER_STEP neighbouring frames averaged."""
    waveform, sr = load_clip(folder, clip)
    features = log_mel(waveform, sr)
    n_steps = features.shape[1] // FRAMES_PER_STEP
    return features[:, :n_steps * FRAMES_PER_STEP].reshape(features.shape[0], n_steps, FRAMES_PER_STEP).mean(axis=2)


def _features_are_current(saved, folder, clips, log_mel):
    # True if the saved features were made with the current log_mel (checked on 3 clips)
    try:
        return all(np.allclose(clip_features(folder, clips[i], log_mel), saved['features'][i], atol=1e-3) for i in range(3))
    except Exception:
        return False


def _compute_features(path, folder, clips, log_mel):
    print("Computing the features of all clips ...")
    start = time.time()
    features = []
    for i, clip in enumerate(clips):
        features.append(clip_features(folder, clip, log_mel))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1} / {len(clips)} clips, {time.time() - start:.0f} s")
    data = {'features': np.stack(features).astype(np.float32),
            'targets': np.array([c['target'] for c in clips]), 'folds': np.array([c['fold'] for c in clips]),
            'categories': np.array([c['category'] for c in clips]), 'filenames': np.array([c['filename'] for c in clips])}
    np.savez(path, **data)
    print(f"Saved the features in {path}")
    return data


def make_datasets(dataset_path, clips, folder, log_mel, autograding=False):
    """Compute (or load) the features of all clips, standardise every mel band, and split by fold.
    Returns train_set, val_set, test_set, train_eval_set (400 training clips) and the class names."""
    path = os.path.join(dataset_path, FEATURES_FILE)
    if autograding:    # the autograder uses the features file made by the TA
        saved = dict(np.load(path))
    else:
        saved = dict(np.load(path)) if os.path.exists(path) else None
        if saved is None or len(saved['features']) != len(clips) or not _features_are_current(saved, folder, clips, log_mel):
            saved = _compute_features(path, folder, clips, log_mel)
    class_names = [str(name) for _, name in sorted(set(zip(saved['targets'].tolist(), saved['categories'].tolist())))]
    inputs = torch.tensor(saved['features'], dtype=torch.float32)
    targets = torch.tensor(saved['targets'], dtype=torch.long)
    folds = torch.tensor(saved['folds'])
    is_train, is_val, is_test = folds <= 3, folds == 4, folds == 5
    band_mean = inputs[is_train].mean(dim=(0, 2), keepdim=True)
    band_std = inputs[is_train].std(dim=(0, 2), keepdim=True)
    inputs = (inputs - band_mean) / band_std
    train_set = TensorDataset(inputs[is_train], targets[is_train])
    val_set = TensorDataset(inputs[is_val], targets[is_val])
    test_set = TensorDataset(inputs[is_test], targets[is_test])
    train_eval_set = TensorDataset(inputs[is_train][::3], targets[is_train][::3])
    print(f"train {len(train_set)}, validation {len(val_set)}, test {len(test_set)} clips; "
          f"input shape {tuple(train_set[0][0].shape)}, {len(class_names)} classes")
    return train_set, val_set, test_set, train_eval_set, class_names


def show_examples(clips, folder, log_mel, categories=('dog', 'rooster', 'rain', 'crying_baby', 'clock_tick', 'church_bells')):
    """Play a clip of each category and show its log-mel spectrogram."""
    from IPython.display import Audio, display
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for ax, category in zip(axes.flat, categories):
        clip = next(c for c in clips if c['category'] == category)
        waveform, sr = load_clip(folder, clip)
        image = ax.imshow(log_mel(waveform, sr), origin='lower', aspect='auto', cmap='magma', extent=[0, 5, 0, 64])
        ax.set(title=category, xlabel='time (s)', ylabel='mel band (low → high)')
        print(category)
        display(Audio(waveform, rate=sr))
    fig.colorbar(image, ax=axes, label='power (dB)')
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Plots

CHANCE_LOSS = np.log(50)    # the loss of a model that guesses: 3.91
LEVELS = [(2.2, 1), (1.9, 2), (1.7, 3)]    # (validation loss, level) of Task 6
session_bests = {}          # the best validation loss of each sweep, for plot_journey()


def format_value(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def plot_history(history, title):
    """Training and validation loss, and validation accuracy, over the epochs."""
    epochs = np.arange(1, len(history['val_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    ax1.plot(epochs, history['train_loss'], label='train')
    ax1.plot(epochs, history['val_loss'], label='validation')
    ax1.fill_between(epochs, history['train_loss'], history['val_loss'], color='tab:red', alpha=0.1, label='generalization gap')
    ax1.axhline(CHANCE_LOSS, color='gray', ls=':', label='chance level')
    ax1.set(xlabel='epoch', ylabel='loss', title=f"{title}: final validation loss {history['val_loss'][-1]:.3f}")
    ax1.set_ylim(0, _ylim(history['val_loss']))
    ax1.legend()
    ax2.plot(epochs, history['val_acc'], color='tab:orange')
    ax2.axhline(1 / 50, color='gray', ls=':', label='chance level')
    ax2.set(xlabel='epoch', ylabel='accuracy', title=f"Validation accuracy: final {history['val_acc'][-1]:.3f}", ylim=(0, 1))
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    plt.show()


RESULT_KEYS = ('train_loss', 'val_loss', 'val_acc', 'time')


def _sweep_figure(grid):
    names = list(grid)
    rows = len(grid[names[0]])
    if len(names) == 1:
        size = (14, 4.5)
    elif len(names) == 2:
        size = (15, 1.5 + 0.75 * rows)
    else:    # one heatmap per slice, PANELS_PER_ROW to a row
        n_slices = len(_slices(names, grid))
        size = (15, 0.8 + -(-n_slices // PANELS_PER_ROW) * (1.0 + 0.45 * rows))
    return plt.figure(figsize=size, constrained_layout=True)


def _best_index(val_loss):
    if not np.isfinite(val_loss).any():
        return None
    return np.unravel_index(np.nanargmin(np.where(np.isfinite(val_loss), val_loss, np.nan)), val_loss.shape)


def _draw_sweep(fig, grid, results, title):
    """Draw sweep results into fig. Runs that have not finished yet are NaN in results."""
    fig.clear()
    names = list(grid)
    val_loss = np.asarray(results['val_loss'], dtype=float)
    best = _best_index(val_loss)
    if len(names) == 1:
        _draw_sweep_1d(fig, names[0], grid[names[0]], results, None if best is None else best[0], title)
    elif len(names) == 2:
        _draw_sweep_2d(fig, names, grid, results, best, title)
    else:
        _draw_sweep_nd(fig, names, grid, results, best, title)
    return best


def _ylim(values):
    finite = np.asarray(values, float)[np.isfinite(values)]
    top = min(finite.max(), 6) if finite.size else 0
    return max(CHANCE_LOSS + 0.3, top) + 0.1


def _draw_sweep_1d(fig, name, values, results, best, title):
    x = np.arange(len(values))
    train_loss, val_loss = np.asarray(results['train_loss'], float), np.asarray(results['val_loss'], float)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.fill_between(x, train_loss, val_loss, color='tab:red', alpha=0.12, label='generalization gap')
    ax1.plot(x, train_loss, 'o-', color='tab:blue', label='train loss')
    ax1.plot(x, val_loss, 'o-', color='tab:orange', label='validation loss')
    if best is not None:
        ax1.plot(x[best], val_loss[best], '*', ms=20, color='gold', mec='black', label=f'best: {name} = {format_value(values[best])}')
    ax1.axhline(CHANCE_LOSS, color='gray', ls=':', label='chance level')
    ax1.set_xticks(x)
    times = np.asarray(results['time'], float)
    ax1.set_xticklabels([f"{format_value(v)}\n{t:.0f} s" if np.isfinite(t) else f"{format_value(v)}\n" for v, t in zip(values, times)])
    ax1.set(xlabel=f"{name}   (below: training time per run)", ylabel='final loss', title=f"{title}: final losses")
    ax1.set_xlim(-0.5, len(values) - 0.5)
    ax1.set_ylim(0, _ylim(val_loss))
    ax1.legend(fontsize=9)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(values)))
    for i, value in enumerate(values):
        history = results['histories'][i]
        if history is None:
            continue
        epochs = np.arange(1, len(history['val_loss']) + 1)
        ax2.plot(epochs, history['val_loss'], color=colors[i], lw=3 if i == best else 1.3,
                 label=f"{name} = {format_value(value)}" + ("  (best)" if i == best else ""))
    ax2.axhline(CHANCE_LOSS, color='gray', ls=':')
    ax2.set(xlabel='epoch', ylabel='validation loss', title='Validation loss during training (first seed)')
    ax2.set_ylim(0, _ylim(val_loss))
    if ax2.lines[:-1]:
        ax2.legend(fontsize=8, ncol=2)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)


def _draw_sweep_2d(fig, names, grid, results, best, title):
    rows, cols = grid[names[0]], grid[names[1]]
    val_loss = np.asarray(results['val_loss'], float)
    gap = val_loss - np.asarray(results['train_loss'], float)
    axes = fig.subplots(1, 2)
    panels = [(val_loss, 'viridis_r', 'final validation loss (bright = better)'),
              (gap, 'Reds', 'generalization gap: validation − train loss')]
    for ax, (data, cmap, label) in zip(axes, panels):
        finite = data[np.isfinite(data)]
        low, high = (finite.min(), finite.max()) if finite.size else (0, 1)
        high = max(high, low + 0.01)    # e.g. after the first run, when all finished cells are equal
        image = ax.imshow(data, cmap=cmap, aspect='auto', vmin=low, vmax=high)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if not np.isfinite(data[i, j]):
                    continue
                shade = (data[i, j] - low) / (high - low)
                dark = shade > 0.55 if cmap == 'viridis_r' else shade > 0.6
                ax.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center', fontsize=9, color='white' if dark else 'black')
        if best is not None:
            ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec='gold', lw=3))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([format_value(v) for v in cols])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([format_value(v) for v in rows])
        ax.set(xlabel=names[1], ylabel=names[0], title=label)
        fig.colorbar(image, ax=ax)
    fig.suptitle(f"{title}   (gold square: lowest validation loss)")


PANELS_PER_ROW = 4


def _slices(names, grid):
    """For three or more hyper-parameters: every combination of the values of the third and later ones."""
    return list(np.ndindex(tuple(len(grid[name]) for name in names[2:])))


def _draw_sweep_nd(fig, names, grid, results, best, title):
    """Three or more hyper-parameters: a heatmap of the first two for every combination of the others (slices of the
    hypercube), all on the same colour scale."""
    val_loss = np.asarray(results['val_loss'], float)
    rows, cols = grid[names[0]], grid[names[1]]
    slices = _slices(names, grid)
    n_cols = min(len(slices), PANELS_PER_ROW)
    n_rows = -(-len(slices) // n_cols)
    finite = val_loss[np.isfinite(val_loss)]
    low, high = (finite.min(), finite.max()) if finite.size else (0, 1)
    high = max(high, low + 0.01)
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    for k, rest in enumerate(slices):
        ax = axes[k // n_cols][k % n_cols]
        data = val_loss[(slice(None), slice(None)) + rest]
        image = ax.imshow(data, cmap='viridis_r', aspect='auto', vmin=low, vmax=high)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if np.isfinite(data[i, j]):
                    dark = (data[i, j] - low) / (high - low) > 0.55
                    ax.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center', fontsize=8, color='white' if dark else 'black')
        if best is not None and tuple(int(b) for b in best[2:]) == rest:
            ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec='gold', lw=3))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([format_value(v) for v in cols], fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([format_value(v) for v in rows], fontsize=8)
        ax.set_xlabel(names[1], fontsize=9)
        if k % n_cols == 0:
            ax.set_ylabel(names[0], fontsize=9)
        ax.set_title(', '.join(f"{name} = {format_value(grid[name][i])}" for name, i in zip(names[2:], rest)), fontsize=10)
    for k in range(len(slices), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis('off')
    fig.colorbar(image, ax=axes, label='final validation loss (bright = better)', shrink=0.9)
    fig.suptitle(f"{title}   (one heatmap per value of {', '.join(names[2:])}; gold square: lowest validation loss)")


def _summarise(grid, results, title, best):
    """Remember the best result of the sweep for plot_journey(), and print it."""
    names = list(grid)
    if best is None:
        print("All runs diverged: there is no best value.")
        return
    val_loss = np.asarray(results['val_loss'], dtype=float)
    best_values = {name: grid[name][i] for name, i in zip(names, best)}
    session_bests[title] = (float(val_loss[best]), best_values)
    print(f"Lowest final validation loss: {val_loss[best]:.3f} with "
          + ", ".join(f"{name} = {format_value(value)}" for name, value in best_values.items()) + ".")


def plot_sweep(grid, results, title):
    """Plot the results of sweep(): a line plot for one hyper-parameter, heatmaps for two, and for three or more a
    heatmap of the first two for every combination of the others."""
    from IPython.display import display
    fig = _sweep_figure(grid)
    best = _draw_sweep(fig, grid, results, title)
    display(fig)
    plt.close(fig)
    _summarise(grid, results, title, best)


# Live progress of a sweep: train_model() calls report_run() after every run, and run_sweep() shows it

_live_sweeps = []


def _duration(seconds):
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"


def _text(text):
    return {'text/plain': text}


class _NoHandle:
    """Stands in for a display handle outside a notebook, where display() returns None."""
    def update(self, *args, **kwargs):
        pass


def _show(obj, raw=False):
    from IPython.display import display
    return display(obj, raw=raw, display_id=True) or _NoHandle()


class _LiveSweep:
    def __init__(self, grid, title, seeds, plot=True, csv_file=None):
        self.grid, self.title, self.seeds = grid, title, tuple(seeds)
        self.csv_file, self.csv_error = csv_file, None
        self.names = list(grid)
        self.shape = tuple(len(grid[name]) for name in self.names)
        self.total = int(np.prod(self.shape)) * len(self.seeds)
        self.sums = {key: np.zeros(self.shape) for key in RESULT_KEYS}
        self.counts = np.zeros(self.shape, dtype=int)
        self.histories = np.empty(self.shape, dtype=object)
        self.done, self.start = 0, time.time()
        self.status = _show(_text(f"{title}: {self.total} training runs to do ..."), raw=True)
        self.plot = _show(_text("(The plot appears after the first run.)"), raw=True) if plot else _NoHandle()
        self.fig = _sweep_figure(grid) if plot else None

    def _index(self, config):
        index = []
        for name in self.names:
            matches = [i for i, value in enumerate(self.grid[name]) if _equal(value, config.get(name))]
            if not matches:
                return None
            index.append(matches[0])
        return tuple(index)

    def add(self, config, seed, history):
        self.done += 1
        index = self._index(config)
        if index is not None:
            for key in RESULT_KEYS:
                self.sums[key][index] += history[key][-1]
            self.counts[index] += 1
            if self.histories[index] is None:
                self.histories[index] = history
            if self.fig is not None:
                partial = {key: np.where(self.counts > 0, self.sums[key] / np.maximum(self.counts, 1), np.nan) for key in RESULT_KEYS}
                partial['histories'] = self.histories
                _draw_sweep(self.fig, self.grid, partial, self.title)
                self.plot.update(self.fig)
        elapsed = time.time() - self.start
        values = ", ".join(f"{name} = {format_value(config.get(name))}" for name in self.names)
        seed_text = f", seed {seed}" if len(self.seeds) > 1 else ""
        left = f"about {_duration(elapsed / self.done * (self.total - self.done))} left" if self.done < self.total else "finishing"
        self.status.update(_text(f"{self.title}: run {self.done} of {self.total} done ({values}{seed_text}): final validation "
                                 f"loss {history['val_loss'][-1]:.3f} in {history['time'][-1]:.0f} s. "
                                 f"Elapsed {_duration(elapsed)}, {left}."), raw=True)

    def write_csv(self, config, seed, history):
        """Append one row for this run to csv_file: the sweep, every hyper-parameter, the seed and the final results."""
        row = {'sweep': self.title, **{key: (value.item() if hasattr(value, 'item') else value) for key, value in config.items()},
               'seed': seed, **{key: round(float(history[key][-1]), 4) for key in ('train_loss', 'val_loss', 'val_acc')},
               'seconds': round(history['time'][-1], 2)}
        try:
            fields = None
            if os.path.exists(self.csv_file) and os.path.getsize(self.csv_file) > 0:
                with open(self.csv_file, newline='') as file:
                    fields = next(csv.reader(file))
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fields or list(row), extrasaction='ignore', restval='')
                if fields is None:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            if self.csv_error is None:
                print(f"Writing to {self.csv_file} failed ({type(e).__name__}: {e}).")
            self.csv_error = e

    def finish(self, results):
        saved = f" Every run is in {self.csv_file}." if self.csv_file and self.csv_error is None else ""
        self.status.update(_text(f"{self.title}: all {self.done} runs done in {_duration(time.time() - self.start)}.{saved}"), raw=True)
        try:
            if self.fig is not None:
                best = _draw_sweep(self.fig, self.grid, results, self.title)
                self.plot.update(self.fig)
            else:
                best = _best_index(np.asarray(results['val_loss'], dtype=float))
            _summarise(self.grid, results, self.title, best)
        except Exception as e:
            print(f"The results could not be {'plotted' if self.fig is not None else 'summarised'} ({type(e).__name__}: {e}). "
                  "They are returned all the same; check that your sweep() returns them in the format of Task 4.")
        finally:
            if self.fig is not None:
                plt.close(self.fig)


def _equal(a, b):
    try:
        return bool(a == b)
    except Exception:
        return False


def report_run(config, seed, history):
    """Called by train_model() after every run. Updates the live plot of run_sweep() and writes its CSV file, if a
    sweep is running."""
    if _live_sweeps:
        live = _live_sweeps[-1]
        if live.csv_file:
            live.write_csv(config, seed, history)
        try:
            live.add(config, seed, history)
        except Exception:
            pass    # the live plot must never stop the training


def run_sweep(sweep, base_config, grid, title, seeds=(0,), plot=True, csv_file=None):
    """Run your sweep(base_config, grid, seeds). While it trains, show its progress and a plot that grows after every
    run; at the end, plot the results that your sweep() returned. Returns those results, also if they cannot be plotted.
    The grid can have one, two, or more hyper-parameters (if your sweep() handles them).

    plot=False: no plots, only the progress and the best result.
    csv_file='my_runs.csv': append one row per run to this file as soon as the run is done (the sweep's title, every
    hyper-parameter, the seed, and the final train loss, validation loss, validation accuracy and run time)."""
    live = _LiveSweep(grid, title, seeds, plot=plot, csv_file=csv_file)
    _live_sweeps.append(live)
    try:
        results = sweep(base_config, grid, seeds=seeds)
    except BaseException:
        if live.fig is not None:
            plt.close(live.fig)
        raise
    finally:
        _live_sweeps.remove(live)
    live.finish(results)
    return results


def plot_journey(base_history, final_history, test_loss):
    """The best validation loss after each session, from the starting configuration to the final model."""
    labels = ['start'] + [title.split(':')[0] for title in session_bests] + ['final model\n(validation)', 'final model\n(test)']
    losses = ([base_history['val_loss'][-1]] + [loss for loss, _ in session_bests.values()]
              + [final_history['val_loss'][-1], test_loss])
    colors = ['gray'] + ['tab:blue'] * len(session_bests) + ['tab:orange', 'tab:red']
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    bars = ax.bar(range(len(losses)), losses, color=colors)
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, loss + 0.03, f"{loss:.2f}", ha='center')
    for threshold, level in LEVELS:
        ax.axhline(threshold, color='green', ls='--', lw=1)
        ax.text(-0.45, threshold + 0.03, f"level {level}: {threshold}", color='green', va='bottom', fontsize=9)
    ax.set_xticks(range(len(losses)))
    ax.set_xticklabels(labels)
    ax.set(ylabel='validation loss', title='Your tuning journey: best validation loss after each session')
    plt.show()


def plot_per_class(model, dataset, class_names, device):
    """Accuracy per class, and the most frequent confusions."""
    model.eval()
    with torch.no_grad():
        inputs, targets = dataset.tensors
        predicted = model(inputs.to(device)).argmax(dim=1).cpu()
    accuracy = np.array([(predicted[targets == c] == c).float().mean().item() for c in range(len(class_names))])
    order = np.argsort(accuracy)
    fig, ax = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    ax.bar(range(len(order)), accuracy[order], color=plt.cm.RdYlGn(accuracy[order]))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([class_names[c] for c in order], rotation=90)
    ax.set(ylabel='accuracy', title=f'Accuracy per class ({len(targets) // len(class_names)} clips per class)')
    plt.show()
    confusions = {}
    for t, p in zip(targets.tolist(), predicted.tolist()):
        if t != p:
            confusions[(t, p)] = confusions.get((t, p), 0) + 1
    print("Most frequent confusions (true class → predicted class):")
    for (t, p), n in sorted(confusions.items(), key=lambda item: -item[1])[:8]:
        print(f"  {class_names[t]:>18s} → {class_names[p]:<18s} {n} clips")


# ---------------------------------------------------------------------------------------------------------------------
# The final configuration

LIMITS = {'n_layers': (1, 5), 'channels': (1, 256), 'kernel_size': (1, 9), 'dropout': (0.0, 0.95),
          'lr': (1e-7, 10.0), 'batch_size': (8, 1200), 'weight_decay': (0.0, 10.0), 'epochs': (1, 100)}
INTEGER_HYPERPARAMETERS = ('n_layers', 'channels', 'kernel_size', 'batch_size', 'epochs')


def checked_config(config):
    """A copy of config with plain Python numbers. Raises an AssertionError if it is outside the limits."""
    assert isinstance(config, dict), f"The configuration should be a dict, but it is {type(config).__name__}."
    missing = [key for key in LIMITS if key not in config]
    assert not missing, f"The configuration has no {', '.join(missing)}."
    clean = {}
    for key, (low, high) in LIMITS.items():
        value = config[key].item() if hasattr(config[key], 'item') else config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            hint = " Did you write a decimal comma (0,4) instead of a decimal point (0.4)?" if isinstance(value, tuple) else ""
            raise AssertionError(f"{key} should be a number, but it is {value!r}.{hint}")
        if key in INTEGER_HYPERPARAMETERS:
            assert float(value) == int(value), f"{key} should be a whole number, but it is {value!r}."
            value = int(value)
        else:
            value = float(value)
        assert low <= value <= high, f"{key} = {value} is outside the allowed range [{low}, {high}]."
        clean[key] = value
    return clean


def saved_model_val_loss(path, model_class, val_set):
    """The validation loss of the model saved in path by the training cell of Task 6, and its configuration."""
    try:
        saved = torch.load(path, map_location='cpu')
    except FileNotFoundError:
        raise AssertionError("best_model.pth was not found. Run the training cell of Task 6, which saves it, "
                             "and submit best_model.pth with the notebook.")
    except Exception as e:
        raise AssertionError(f"best_model.pth could not be read ({type(e).__name__}). Save it with the training cell of Task 6.")
    assert isinstance(saved, dict) and 'config' in saved and 'state_dict' in saved, \
        "best_model.pth should contain the configuration and the weights. Save it with the training cell of Task 6."
    config = checked_config(saved['config'])
    model = model_class(config['n_layers'], config['channels'], config['kernel_size'], config['dropout'])
    try:
        model.load_state_dict(saved['state_dict'])
    except Exception as e:
        raise AssertionError(f"The weights in best_model.pth do not fit SoundNet with the saved configuration "
                             f"({(str(e).strip().splitlines() or [''])[0][:200]}). Did you change SoundNet after training? "
                             "Run the training cell of Task 6 again.")
    model.eval()
    with torch.no_grad():
        inputs, targets = val_set.tensors
        loss = nn.functional.cross_entropy(model(inputs.cpu()), targets.cpu()).item()
    assert np.isfinite(loss), "The validation loss of your saved model is not a finite number: the training diverged."
    return loss, config


# ---------------------------------------------------------------------------------------------------------------------
# Test helpers

@contextmanager
def in_scratch_folder():
    """Run code in a temporary folder, so that files it writes are thrown away."""
    folder, scratch = os.getcwd(), tempfile.mkdtemp()
    os.chdir(scratch)
    try:
        yield
    finally:
        os.chdir(folder)
        shutil.rmtree(scratch, ignore_errors=True)


def as_float(value, what):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise AssertionError(f"{what} should be a number, but it is {type(value).__name__}.")


def check_answer_format(answer, name):
    """The answer to a multiple-choice question, lower case. Raises an AssertionError if it is not 'a', 'b' or 'c'."""
    assert answer is not None, (f"{name} is not set. Set it to 'a', 'b' or 'c' in its cell, and check that the cell runs "
                               "without errors.")
    assert isinstance(answer, str) and answer.strip().lower() in ('a', 'b', 'c'), \
        f"{name} should be 'a', 'b' or 'c', but it is {answer!r}."
    return answer.strip().lower()


# Task 1

def test_signal(seconds, sr, seed=0):
    """A rising tone plus a little noise."""
    t = np.arange(int(seconds * sr)) / sr
    noise = np.random.default_rng(seed).normal(0, 0.01, len(t))
    return (0.5 * np.sin(2 * np.pi * (200 + 1500 * t) * t) + noise).astype(np.float32)


def check_log_mel(log_mel, waveform, sr):
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
    expected = librosa.power_to_db(mel)
    result = log_mel(waveform, sr)
    assert isinstance(result, np.ndarray), f"log_mel() should return a numpy array, but it returned {type(result).__name__}."
    if result.shape != expected.shape:
        if result.shape == expected.shape[::-1]:
            raise AssertionError(f"log_mel() returned shape {result.shape}, which is transposed: it should be {expected.shape}, "
                                 "(mel bands, frames), as librosa returns it.")
        hints = []
        if result.ndim == 2 and result.shape[0] != 64:
            hints.append("Use n_mels=64.")
        if result.ndim == 2 and result.shape[1] != expected.shape[1]:
            hints.append("Use hop_length=512.")
        raise AssertionError(f"For a {len(waveform) / sr:g} s signal at {sr} Hz, log_mel() should return shape {expected.shape}, "
                             f"but it returned {result.shape}. " + " ".join(hints))
    if np.allclose(result, expected, atol=1e-3):
        return
    if np.allclose(result, mel, rtol=1e-3, atol=1e-6):
        raise AssertionError("log_mel() returns the power, not decibels. Convert it with librosa.power_to_db.")
    if np.allclose(result, librosa.power_to_db(mel, ref=np.max), atol=1e-3):
        raise AssertionError("Use librosa.power_to_db with its default arguments (ref=1.0), not ref=np.max.")
    raise AssertionError("The values returned by log_mel() are not the expected log-mel spectrogram. "
                         "Check that you pass sr and use n_fft=1024, hop_length=512 and n_mels=64.")


# Task 2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expected_parameters(n_layers, channels, kernel_size, n_mels=64, n_classes=50):
    total, in_channels = 0, n_mels
    for _ in range(n_layers):
        total += in_channels * channels * kernel_size + channels + 2 * channels    # convolution and batch norm
        in_channels = channels
    return total + channels * n_classes + n_classes


def check_model(model_class, n_layers, channels, kernel_size, dropout, n_mels=64, steps=54):
    """Build model_class with these hyper-parameters and check its layers, parameters and output shape."""
    model = model_class(n_layers, channels, kernel_size, dropout, n_mels=n_mels)
    counts = {kind.__name__: sum(isinstance(m, kind) for m in model.modules())
              for kind in (nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool1d)}
    wrong = [f"{count} {name} layers" for name, count in counts.items() if count != n_layers]
    assert not wrong, (f"With n_layers={n_layers}, there should be {n_layers} layers of each kind in the blocks, "
                       f"but there are {', '.join(wrong)}.")
    n_params, expected = count_parameters(model), expected_parameters(n_layers, channels, kernel_size, n_mels)
    assert n_params == expected, (f"SoundNet(n_layers={n_layers}, channels={channels}, kernel_size={kernel_size}) should have "
                                  f"{expected} parameters, but it has {n_params}. Check the channels of the convolutions.")
    model.eval()
    try:
        with torch.no_grad():
            output = model(torch.randn(3, n_mels, steps))
    except Exception as e:
        raise AssertionError(f"Passing an input of shape (3, {n_mels}, {steps}) through SoundNet(n_layers={n_layers}, "
                             f"channels={channels}, kernel_size={kernel_size}) raised {type(e).__name__}: {e}")
    assert tuple(output.shape) == (3, 50), f"The output should have shape (3, 50), but it has shape {tuple(output.shape)}."
    return model


# Task 3

class StandIn(nn.Module):
    """A tiny fixed model with batch normalization and dropout, used to test train_one_epoch() and evaluate()."""
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(6)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(6, 3)
        with torch.no_grad():
            self.linear.weight.copy_(torch.arange(18.).view(3, 6).sin())
            self.linear.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        self.inputs, self.training_modes, self.grad_enabled = [], [], []

    def forward(self, x):
        self.inputs.append(x.detach().cpu().clone())
        self.training_modes.append(self.training)
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.linear(self.dropout(self.norm(x)))


def stand_in_model_and_data(device):
    """The stand-in model and 10 fixed samples (6 features, 3 classes) in batches of 4, 4 and 2.
    The last batch has a clearly different loss, so that a wrong average is noticed."""
    inputs = torch.randn(10, 6, generator=torch.Generator().manual_seed(0))
    inputs[8:] *= 4
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 2, 1, 0, 0])
    return StandIn().to(device), DataLoader(TensorDataset(inputs, targets), batch_size=4)


class RecordingLoss(nn.CrossEntropyLoss):
    """Cross-entropy that records the loss and the batch size of every call."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, outputs, targets):
        loss = super().forward(outputs, targets)
        self.calls.append((loss.item(), len(targets)))
        return loss


def check_average(result, calls, what):
    """Accept the average per sample and, if the student averaged the batch means, that too."""
    result = as_float(result, what)
    assert calls, "loss_fn was never called. Compute the loss of every batch with loss_fn(outputs, targets)."
    per_sample = sum(loss * n for loss, n in calls) / sum(n for _, n in calls)
    per_batch = sum(loss for loss, _ in calls) / len(calls)
    assert abs(result - per_sample) < 1e-4 or abs(result - per_batch) < 1e-4, \
        f"{what} is {result:.4f}, but the average loss per sample over the batches is {per_sample:.4f}."


def check_train_one_epoch(train_one_epoch, device):
    stand_in, loader = stand_in_model_and_data(device)
    optimizer = torch.optim.SGD(stand_in.parameters(), lr=0.1)
    weights_before = [p.detach().clone() for p in stand_in.parameters()]
    loss_fn = RecordingLoss()
    with in_scratch_folder():
        result = train_one_epoch(stand_in, loader, optimizer, loss_fn)
    assert len(stand_in.inputs) == 3, f"There are 3 batches, but the model was called {len(stand_in.inputs)} times."
    assert not all(torch.equal(b, p.detach()) for b, p in zip(weights_before, stand_in.parameters())), \
        "The weights did not change. Compute the gradients with loss.backward() and update the weights with optimizer.step()."
    check_average(result, loss_fn.calls, "The value returned by train_one_epoch()")


def expected_evaluation(model, loader, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = (torch.cat(t) for t in zip(*loader))
        outputs = model(inputs.to(device))
        loss = nn.functional.cross_entropy(outputs, targets.to(device)).item()
        accuracy = (outputs.argmax(dim=1).cpu() == targets).float().mean().item()
    return loss, accuracy


def check_evaluate(evaluate, device):
    stand_in, loader = stand_in_model_and_data(device)
    stand_in.train()
    loss_fn = RecordingLoss()
    with in_scratch_folder():
        result = evaluate(stand_in, loader, loss_fn)
    assert isinstance(result, (tuple, list)) and len(result) == 2, "evaluate() should return two values: (loss, accuracy)."
    loss = as_float(result[0], "The loss returned by evaluate()")
    accuracy = as_float(result[1], "The accuracy returned by evaluate()")
    expected_loss, expected_accuracy = expected_evaluation(StandIn().to(device), loader, device)
    check_average(loss, loss_fn.calls, "The loss returned by evaluate()")
    assert abs(loss - expected_loss) < 1e-4, (f"The loss returned by evaluate() is {loss:.4f}, but it should be {expected_loss:.4f}. "
                                              "Did you set the model to evaluation mode?")
    assert abs(accuracy - expected_accuracy) < 1e-6, \
        f"The accuracy returned by evaluate() is {accuracy:.4f}, but it should be {expected_accuracy:.4f} (the fraction of correct predictions)."


# Task 4

FAKE_BASE_CONFIG = {'alpha': 1, 'beta': 'x', 'gamma': 0.5, 'n_layers': 'not a number'}


def fake_score(config, seed):
    return 10 * config['alpha'] + {'x': 1, 'y': 2, 'z': 3}[config['beta']] + config['gamma'] + 0.1 * seed


class FakeTrainModel:
    """Stands in for train_model() in the sweep() tests: it returns a made-up history at once and records its calls."""
    def __init__(self):
        self.calls = []

    def __call__(self, config, seed=0, verbose=False):
        missing = [key for key in FAKE_BASE_CONFIG if key not in config]
        assert not missing, (f"sweep() trained a configuration without {', '.join(missing)}. Every configuration should "
                             "contain all hyper-parameters of base_config, with the values of the grid changed.")
        self.calls.append((copy.deepcopy(config), seed))
        score = fake_score(config, seed)
        history = {'train_loss': [score + 5, score - 1.5, score - 1],
                   'val_loss': [score + 6, score - 0.5, score],    # the last epoch is not the best one
                   'val_acc': [0.1, 0.9, score / 100],
                   'time': [1.0, 2.0, 3.0 + seed]}
        return None, history


def run_fake_sweep(sweep, grid, seeds=None):
    """Run sweep() with train_model() replaced by FakeTrainModel. Returns the results and the recorded calls."""
    fake = FakeTrainModel()
    base = copy.deepcopy(FAKE_BASE_CONFIG)
    namespace = sweep.__globals__    # the notebook, where sweep() finds train_model()
    missing = object()
    original = namespace.get('train_model', missing)
    namespace['train_model'] = fake
    try:
        with in_scratch_folder():
            results = sweep(base, grid) if seeds is None else sweep(base, grid, seeds=seeds)
    except NotImplementedError:
        raise
    except Exception as e:
        if fake.calls:
            raise
        raise AssertionError("sweep() should train every configuration with train_model(config, seed=seed). The test replaces "
                             "train_model() with a fake that returns a made-up history at once, but your sweep() did not call "
                             f"it (it raised {type(e).__name__}: {str(e)[:100]}). Call train_model directly, not through a "
                             "default argument or another name.") from e
    finally:
        if original is missing:
            del namespace['train_model']
        else:
            namespace['train_model'] = original
    assert fake.calls, "sweep() should train every configuration with train_model(config, seed=seed)."
    assert base == FAKE_BASE_CONFIG, "sweep() changed base_config. Change a copy instead (copy.deepcopy(base_config))."
    assert isinstance(results, dict), f"sweep() should return a dict, but it returned {type(results).__name__}."
    missing = [key for key in ('train_loss', 'val_loss', 'val_acc', 'time', 'histories') if key not in results]
    assert not missing, f"The dict returned by sweep() has no {', '.join(repr(k) for k in missing)}."
    return results, fake.calls


def _expected_value(key, config, seeds):
    scores = [fake_score(config, seed) for seed in seeds]
    final = {'train_loss': [s - 1 for s in scores], 'val_loss': scores, 'val_acc': [s / 100 for s in scores],
             'time': [3.0 + seed for seed in seeds]}[key]
    return float(np.mean(final))


def check_sweep_results(results, grid, seeds):
    """Compare the results of a fake sweep with the expected values."""
    names = list(grid)
    shape = tuple(len(grid[name]) for name in names)
    for key in ('train_loss', 'val_loss', 'val_acc', 'time'):
        values = np.asarray(results[key], dtype=float)
        assert values.shape == shape, f"results[{key!r}] should have shape {shape}, but it has shape {values.shape}."
        for index in np.ndindex(shape):
            config = dict(FAKE_BASE_CONFIG, **{name: grid[name][i] for name, i in zip(names, index)})
            expected = _expected_value(key, config, seeds)
            where = ", ".join(f"{name} = {grid[name][i]!r}" for name, i in zip(names, index))
            if abs(values[index] - expected) > 1e-6:
                if key == 'val_loss' and abs(values[index] - (expected - 0.5)) < 1e-6:
                    raise AssertionError("results['val_loss'] holds the lowest validation loss during training. "
                                         "It should hold the final one, at the last epoch.")
                raise AssertionError(f"results[{key!r}] for {where} is {values[index]:g}, but it should be {expected:g}"
                                     + (f", the final value averaged over the seeds {tuple(seeds)}." if len(seeds) > 1 else
                                        ", the final value of the history (at the last epoch)."))
    histories = np.asarray(results['histories'], dtype=object)
    assert histories.shape == shape, f"results['histories'] should have shape {shape}, but it has shape {histories.shape}."
    for index in np.ndindex(shape):
        config = dict(FAKE_BASE_CONFIG, **{name: grid[name][i] for name, i in zip(names, index)})
        history = histories[index]
        assert isinstance(history, dict) and abs(history['val_loss'][-1] - fake_score(config, seeds[0])) < 1e-9 \
            and abs(history['time'][-1] - (3.0 + seeds[0])) < 1e-9, \
            "results['histories'] should hold, for every combination, the history of the first seed."


def check_sweep(sweep):
    grid = {'alpha': [3, 1, 2]}
    results, calls = run_fake_sweep(sweep, grid)
    assert len(calls) == 3, f"With 3 values and one seed, sweep() should train 3 models, but it trained {len(calls)}."
    assert sorted(c['alpha'] for c, _ in calls) == [1, 2, 3], "Each value of the grid should be trained once."
    for config, _ in calls:
        assert all(config[k] == v for k, v in FAKE_BASE_CONFIG.items() if k != 'alpha'), \
            "The other hyper-parameters of the trained configurations should be those of base_config."
    check_sweep_results(results, grid, (0,))
