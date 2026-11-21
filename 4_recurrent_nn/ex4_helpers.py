"""Helpers for Exercise 4: the code behind the tests, and what the notebook needs for grading.

You do not need to read or change this file to do the exercise. Keep it in the same folder as ex4.ipynb.
At grading, the original version of this file is used.
"""
import os
import re
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

TEXT_LENGTH = 135002     # characters in the cleaned text of Alice's Adventures in Wonderland

# The scored tests, in the order of the notebook, with their points
SCORED_TESTS = [
    ('Task 2: character encoding (visible test)', 1),
    ('Task 2: character encoding (hidden test)', 1),
    ('Task 3: get_batches() without overlap (visible test)', 2),
    ('Task 3: get_batches() with overlap (visible test)', 2),
    ('Task 3: get_batches() step size and targets (hidden test)', 2),
    ('Task 4: dropout (visible test)', 1),
    ('Task 4: shapes of the output and the hidden state (visible test)', 2),
    ('Task 4: number of parameters (hidden test)', 3),
    ('Task 5: what train() gives the loss function (visible test)', 1),
    ('Task 5: the loss of your saved model (hidden test)', 2),
    ('Task 6: one-hot input and the last time step (hidden test)', 1),
    ('Task 6: one model call per generated character (hidden test)', 1),
    ('Task 6: temperature (hidden test)', 1),
]

_results = {}               # test name -> None if it passed, else the reason
_submitted_model = None     # the bytes of best_model.pth, read at the start of grading


# ---------------------------------------------------------------------------------------------------------------------
# Grading

def start_grading(autograding=False):
    """True while the autograder grades the notebook. Then it also keeps a copy of best_model.pth, read before any of
    your code runs, so that the last test grades exactly the file you submitted."""
    global _submitted_model
    grading = os.environ.get('NBGRADER_EXECUTION') == 'autograde' or bool(autograding)
    if grading and os.path.isfile('best_model.pth'):
        with open('best_model.pth', 'rb') as f:
            _submitted_model = f.read()
    return grading


def submitted_model():
    """The bytes of the submitted best_model.pth (the copy taken at the start of grading), or None if there is none."""
    if _submitted_model is not None:
        return _submitted_model
    if os.path.isfile('best_model.pth'):
        with open('best_model.pth', 'rb') as f:
            return f.read()
    return None


def training_allowed(skip_training):
    """Training runs only when you run the notebook yourself: never while nbgrader grades or validates it."""
    return not skip_training and not os.environ.get('NBGRADER_EXECUTION')


def text_path(txt_path, ta_txt_path=None, grading=False):
    """The text file to read: during grading, the TA's copy of the same text; otherwise txt_path."""
    folder = os.environ.get('EX4_DATASET_PATH')
    if folder:
        return os.path.join(folder, 'alice.txt')
    if ta_txt_path:
        return ta_txt_path
    if grading:
        raise RuntimeError('TA: no text for grading. Set the environment variable EX4_DATASET_PATH to the folder that '
                           'holds alice.txt, or TA_txt_path in the hidden cell at the top of the source notebook.')
    return txt_path


def run_test(name, check):
    """Run one test. check() raises an error if something is wrong; the result also goes into the feedback file."""
    try:
        try:
            check()
        except NameError as error:
            tb = error.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            frame = tb.tb_frame
            while frame.f_back and frame.f_code.co_name in ('<listcomp>', '<genexpr>', '<dictcomp>', '<setcomp>'):
                frame = frame.f_back
            if not (frame.f_code.co_name == '<lambda>' or frame.f_code.co_name.startswith('test_')):
                raise                   # a name missing inside the student's own code: keep the error as it is
            # a name the test itself needs was never defined: say why, not just the name
            missing = re.search(r"name '(\w+)' is not defined", str(error))
            missing = missing.group(1) if missing else str(error)
            raise AssertionError(f'{missing} is not defined, so this test could not run: the cell that should define it '
                                 f'raised an error (see that cell) or was not run.') from None
    except BaseException as error:
        _results[name] = str(error) if isinstance(error, AssertionError) else f'{type(error).__name__}: {error}'
        raise
    _results[name] = None
    print(f'\033[92m{name}: passed\033[0m')


def write_feedback(path, ta_name):
    """The feedback file: every scored test that failed or did not run, and the 0-point checks that failed."""
    lines = []
    for name, points in SCORED_TESTS:
        if name not in _results:
            lines.append(f'{name}, {points} point(s): did not run (its cell was deleted, or did not get that far).')
        elif _results[name] is not None:
            lines.append(f'{name}, {points} point(s): {_results[name]}')
    scored = {name for name, _ in SCORED_TESTS}
    for name, reason in _results.items():
        if name not in scored and reason is not None:
            lines.append(f'{name}: {reason}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n' if lines else 'No problems found. Good job!\n')
        f.write(f'Contact your TA if you need more feedback. Responsible TA: {ta_name}\n')


# ---------------------------------------------------------------------------------------------------------------------
# Visible tests

def check_text_cleaning(raw_text, cleaned_text):
    errors = []
    if not len(cleaned_text) < len(raw_text):
        errors.append('The cleaned text should be shorter than the raw text.')
    if not cleaned_text.islower():
        errors.append('The cleaned text is not all lowercase.')
    if not all(ch.isalnum() or ch == ' ' for ch in cleaned_text):
        errors.append('The cleaned text still has characters other than letters, digits and spaces.')
    if '  ' in cleaned_text:
        errors.append('The cleaned text has several spaces in a row.')
    assert not errors, ' '.join(errors)


def check_encoding(cleaned_text, encoded_chars, char_to_int, int_to_char):
    errors = []
    if len(char_to_int) != len(int_to_char):
        errors.append('char_to_int and int_to_char should have the same length.')
    if len(encoded_chars) != len(cleaned_text):
        errors.append('encoded_chars should be as long as cleaned_text.')
    try:
        if ''.join(int_to_char[i] for i in encoded_chars) != cleaned_text:
            errors.append('Decoding encoded_chars with int_to_char does not give cleaned_text back.')
    except Exception as e:
        errors.append(f'Decoding encoded_chars with int_to_char failed: {type(e).__name__}: {e}')
    assert not errors, ' '.join(errors)
    if sorted(char_to_int, key=char_to_int.get) != sorted(char_to_int):
        print('Note: your char_to_int does not follow the sorted order of the characters. The order of set() changes '
              'every time Python starts, so a model saved in one session does not fit the mapping of the next one, '
              'nor the mapping at grading. Sort the characters.')


def check_batch_shapes(get_batches, step_size=None):
    """get_batches() on an array as long as the cleaned text (135002 characters), batch_size 64, seq_length 100."""
    arr = np.arange(TEXT_LENGTH) % 27
    if step_size is None:
        result, call = get_batches(arr, batch_size=64, seq_length=100), 'get_batches(arr, batch_size=64, seq_length=100)'
    else:
        result = get_batches(arr, batch_size=64, seq_length=100, step_size=step_size)
        call = f'get_batches(arr, batch_size=64, seq_length=100, step_size={step_size})'
    assert isinstance(result, tuple) and len(result) == 2, 'get_batches() should return two arrays: x_batches, y_batches.'
    x_batches, y_batches = (np.asarray(r) for r in result)
    num_batches = (TEXT_LENGTH - 100) // (step_size or 100) // 64
    expected = (num_batches, 64, 100)
    errors = [f'{name} has shape {tuple(b.shape)}' for name, b in (('x_batches', x_batches), ('y_batches', y_batches))
              if tuple(b.shape) != expected]
    assert not errors, (f'For an array of {TEXT_LENGTH} characters (as long as the cleaned text), {call} should give '
                        f'x_batches and y_batches of shape {expected}, but ' + ' and '.join(errors) + '.')


def check_dropout(CharLSTM, device):
    model = CharLSTM(num_layers=2, input_dim=10, hidden_dim=100, output_dim=40, dropout_prob=0.1).to(device)
    hidden = model.init_hidden(64)
    x = torch.rand(64, 50, 10, device=device)
    model.train()
    out1, _ = model(x, hidden)
    out2, _ = model(x, hidden)
    model.eval()
    out3, _ = model(x, hidden)
    out4, _ = model(x, hidden)
    errors = []
    if torch.equal(out1, out2):
        errors.append('In training mode, two forward passes give the same output: dropout has no effect.')
    if not torch.equal(out3, out4):
        errors.append('In evaluation mode, two forward passes should give the same output.')
    assert not errors, ' '.join(errors)


def check_lstm_shapes(CharLSTM, device):
    model = CharLSTM(num_layers=2, input_dim=10, hidden_dim=100, output_dim=40, dropout_prob=0.1).to(device)
    hidden = model.init_hidden(64)
    x = torch.rand(64, 50, 10, device=device)
    try:
        output, (h, c) = model(x, hidden)
    except RuntimeError as e:
        raise AssertionError(f'The forward pass failed: {e}. Check batch_first=True in nn.LSTM, and that init_hidden() '
                             f'returns states of shape (num_layers, batch_size, hidden_dim).') from None
    errors = [f'{name} should have shape {want}, but has {tuple(t.shape)}.' for name, t, want in
              (('The output', output, (64, 50, 40)), ('h', h, (2, 64, 100)), ('c', c, (2, 64, 100)))
              if tuple(t.shape) != want]
    assert not errors, 'CharLSTM(num_layers=2, input_dim=10, hidden_dim=100, output_dim=40) on a batch of 64 ' \
                       'sequences of 50 steps: ' + ' '.join(errors)


class _RecordingLoss:
    """Stands in for the loss function in train() (criterion, F.cross_entropy or F.nll_loss): records the shapes it
    gets and returns 0."""

    def __init__(self):
        self.calls = []

    def __call__(self, output, target, *args, **kwargs):
        self.calls.append((tuple(output.shape), tuple(target.shape)))
        return torch.zeros((), requires_grad=True, device=output.device)


@contextmanager
def _in_scratch_folder():
    """Run in an empty temporary folder, so that a train() that saves files cannot overwrite your best_model.pth."""
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(here)


def _run_train(train, CharLSTM, device, vocab=50, batch_size=2, seq_length=3, num_batches=11):
    """Run train() for one epoch on small random batches, with a recording loss and a spy on the model's forward()."""
    g = np.random.default_rng(0)
    batches = (g.integers(0, vocab, (num_batches, batch_size, seq_length)).astype(np.int32),
               g.integers(0, vocab, (num_batches, batch_size, seq_length)).astype(np.int32))
    model = CharLSTM(2, vocab, 12, vocab, 0.1).to(device)
    forward_calls = []
    forward = model.forward

    def spy(x, hidden):
        forward_calls.append((tuple(x.shape), x.dtype, bool(hidden[0].requires_grad or hidden[1].requires_grad)))
        return forward(x, hidden)
    model.forward = spy
    loss = _RecordingLoss()
    replaced = {'tqdm': lambda iterable, *a, **k: iterable, 'get_batches': lambda *a, **k: batches}
    with patch.dict(train.__globals__, replaced), patch('torch.nn.CrossEntropyLoss', return_value=loss), \
            patch('torch.nn.functional.cross_entropy', side_effect=loss), \
            patch('torch.nn.functional.nll_loss', side_effect=loss), _in_scratch_folder():
        train(model, np.arange(vocab), vocab, num_epochs=1, batch_size=batch_size, seq_length=seq_length,
              step_size=seq_length, learning_rate=0.001, verbose=False)
    return forward_calls, loss.calls, num_batches


def check_train_basics(train, CharLSTM, device):
    """0 points: the model gets one-hot input and detached hidden states."""
    forward_calls, _, n = _run_train(train, CharLSTM, device)
    assert forward_calls, 'train() never called the model.'
    errors = []
    shapes = {s for s, _, _ in forward_calls}
    if shapes != {(2, 3, 50)}:
        errors.append(f'With batch_size 2, seq_length 3 and 50 characters, the model input should have shape '
                      f'(2, 3, 50) (one-hot), but it had {sorted(shapes)}.')
    if any(not d.is_floating_point for _, d, _ in forward_calls):
        errors.append('The one-hot input should be a float tensor (.float()).')
    if any(r for _, _, r in forward_calls[1:]):
        errors.append('The hidden states given to the model should be detached from the previous batch.')
    assert not errors, ' '.join(errors)


def check_train_loss(train, CharLSTM, device):
    """1 point: what train() gives the loss function, in every batch."""
    _, loss_calls, n = _run_train(train, CharLSTM, device)
    assert loss_calls, 'train() did not compute the loss with criterion, the loss function defined in train().'
    assert len(loss_calls) >= n, f'train() should compute the loss in every batch: {n} batches, but {len(loss_calls)} loss(es).'
    wrong = sorted({c for c in loss_calls if c != ((6, 50), (6,))})
    assert not wrong, (f'With batch_size 2, seq_length 3 and 50 characters, criterion should get the output of shape '
                       f'(6, 50) and the targets of shape (6,), but it got {wrong[0][0]} and {wrong[0][1]}.')


class _StopGeneration(Exception):
    pass


def spy_generate_text(generate_text, CharLSTM, char_to_int, int_to_char, device, temperature=1.0, predict_len=1,
                      stop_at_softmax=False):
    """Run generate_text() with a small untrained CharLSTM (always the same one) and record what happens: the model's
    inputs, the last time step of each model output, the inputs of softmax, and the generated text or the error."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    np.random.seed(0)          # generate_text() samples with np.random.choice
    vocab = len(char_to_int)
    model = CharLSTM(2, vocab, 20, vocab, 0.2).to(device)
    record = {'inputs': [], 'last': [], 'softmax': [], 'text': None, 'error': None}
    forward, softmax = model.forward, F.softmax
    in_model = [False]

    def forward_spy(x, hidden):
        in_model[0] = True
        try:
            out, h = forward(x, hidden)
        finally:
            in_model[0] = False
        record['inputs'].append(x.detach().clone().cpu())
        record['last'].append((out[:, -1, :] if out.dim() == 3 else out).detach().clone().cpu())
        return out, h

    def softmax_spy(t, *args, **kwargs):
        if in_model[0]:                 # a softmax inside the model's forward() is not generate_text()'s
            return softmax(t, *args, **kwargs)
        record['softmax'].append(t.detach().clone().cpu())
        if stop_at_softmax:
            raise _StopGeneration()
        return softmax(t, *args, **kwargs)

    model.forward = forward_spy
    with patch('torch.nn.functional.softmax', side_effect=softmax_spy):
        try:
            record['text'] = generate_text(model, 'we re all ', char_to_int, int_to_char, vocab,
                                           predict_len=predict_len, temperature=temperature)
        except _StopGeneration:
            pass
        except Exception as e:
            record['error'] = e
    return record
