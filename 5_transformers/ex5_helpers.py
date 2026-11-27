"""Helpers for Exercise 5: the code behind the tests, and what the notebook needs for grading.

You do not need to read or change this file to do the exercise. Keep it in the same folder as ex5.ipynb.
At grading, the original version of this file is used.
"""
import io
import math
import os
import re
import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

# The model of Task 3, as the locked cell of Task 3 builds it (the vocabulary sizes come from your vocabularies)
EMBED_SIZE, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, MAX_LEN = 240, 6, 512, 4, 256
LOSS_LIMIT = 0.2           # the largest validation loss of model.pth that scores the Task 3 points (correct training:
                           # 0.05 to 0.14 after 3 epochs, about 0.15 after 1; broken training: above 1)
SUBSET = 2048              # Task 3 is graded on the first 2048 validation sentences: all 27,392 take a minute on a CPU
PAD, SOS, EOS = 0, 1, 2

# The scored tests, in the order of the notebook, with their points
SCORED_TESTS = [
    ('Task 1.1: tokenize() (visible test)', 1),
    ('Task 1.2: build_vocab() (visible test)', 1),
    ('Task 1.3: the embedding layers (hidden test)', 1),
    ('Task 1.4: positional encoding (visible test)', 2),
    ('Task 2.1: MySimpleTransformer (visible test)', 3),
    ('Task 2.2: MyTransformer (visible test)', 2),
    ('Task 3: the validation loss of your saved model.pth (hidden test)', 5),
    ('Task 4: the translation loop (hidden test)', 5),
]

_results = {}               # test name -> None if it passed, else the reason
_submitted = {}             # file name -> its bytes, read at the start of grading


# ---------------------------------------------------------------------------------------------------------------------
# Grading

@contextmanager
def _quiet():
    """No gradients and no warnings: nbgrader counts a warning (output on stderr) as a failed test, and torch warns
    about its own internals in evaluation mode."""
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield

def start_grading(autograding=False):
    """True while the autograder grades the notebook. Then it also keeps a copy of model.pth, read before any of your
    code runs, so that the tests grade exactly the file you submitted."""
    grading = os.environ.get('NBGRADER_EXECUTION') == 'autograde' or bool(autograding)
    if grading and os.path.isfile('model.pth'):
        with open('model.pth', 'rb') as f:
            _submitted['model.pth'] = f.read()
    return grading


def submitted_model():
    """The bytes of the submitted model.pth (the copy taken at the start of grading), or None if there is none."""
    if 'model.pth' in _submitted:
        return _submitted['model.pth']
    if os.path.isfile('model.pth'):
        with open('model.pth', 'rb') as f:
            return f.read()
    return None


def training_allowed(skip_training):
    """Training runs only when you run the notebook yourself: never while nbgrader grades or validates it."""
    return not skip_training and not os.environ.get('NBGRADER_EXECUTION')


def data_folder(path, ta_data_path=None, grading=False):
    """The folder with small_vocab_en.csv and small_vocab_fr.csv: during grading, the TA's copy of the same files;
    otherwise path."""
    folder = os.environ.get('EX5_DATASET_PATH') or ta_data_path
    if not folder and grading:
        raise RuntimeError('TA: no data for grading. Set the environment variable EX5_DATASET_PATH to the folder that '
                           'holds small_vocab_en.csv and small_vocab_fr.csv (the repo has one: '
                           'Ex5_Transformers/dataset_ex5), or TA_data_path in the hidden cell at the top of the notebook.')
    folder = folder or path
    for name in ('small_vocab_en.csv', 'small_vocab_fr.csv'):
        if not os.path.isfile(os.path.join(folder, name)):
            raise FileNotFoundError(f'{name} was not found in {os.path.abspath(folder)}. Set path (above) to the folder '
                                    f'that holds small_vocab_en.csv and small_vocab_fr.csv: dataset_ex5, which comes '
                                    f'with this notebook. Check for an extra folder level after unzipping.')
    return folder


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
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n' if lines else 'No problems found. Good job!\n')
        f.write(f'Contact your TA if you need more feedback. Responsible TA: {ta_name}\n')


# ---------------------------------------------------------------------------------------------------------------------
# Visible tests of Task 1

def check_tokenize(tokenize):
    sentences = ["Hello, world!", "Python is fun.", "Let's tokenize this: right?",
                 "Il a vu «la» tour Eiffel l'été dernier."]
    expected = [["hello", "world"], ["python", "is", "fun"], ["let's", "tokenize", "this", "right"],
                ["il", "a", "vu", "la", "tour", "eiffel", "l'été", "dernier"]]
    got = tokenize(sentences)
    try:
        got = [list(s) for s in got]
    except TypeError:
        raise AssertionError(f'tokenize() should return a list of token lists, but it returned {got!r}.') from None
    wrong = [f'{s!r} gave {g}, expected {e}' for s, g, e in zip(sentences, got, expected) if g != e]
    if len(got) != len(expected):
        wrong.insert(0, f'{len(sentences)} sentences gave {len(got)} token lists')
    if not wrong:
        return
    tokens = [t for s in got for t in s if isinstance(t, str)]
    filters = set('.?!#$%&()*+,-/:;<=>@«»""[\\]^_`{|}~\t\n')
    problems = []
    if any(t != t.lower() for t in tokens):
        problems.append('the tokens are not lowercased')
    if any(ch in filters for t in tokens for ch in t):
        problems.append('characters of filters are left in the tokens')
    if not any("'" in t for t in tokens) or not any('é' in t for t in tokens):
        problems.append('characters that are not in filters were removed (keep apostrophes and accents)')
    reason = ' and '.join(problems) or 'the tokens are not as expected'
    raise AssertionError(f'tokenize(): {reason}. ' + '; '.join(wrong) + '.')


def check_vocab(build_vocab):
    sample = [["hello", "world"], ["hello", "my", "friend"]]
    expected = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "friend": 3, "hello": 4, "my": 5, "world": 6}
    result = build_vocab(sample)
    assert isinstance(result, tuple) and len(result) == 2, 'build_vocab() should return two dictionaries: word2idx, idx2word.'
    word2idx, idx2word = result
    assert isinstance(word2idx, dict) and isinstance(idx2word, dict), (
        f'build_vocab() should return two dictionaries, but it returned a {type(word2idx).__name__} and a '
        f'{type(idx2word).__name__}.')
    if dict(word2idx) != expected:
        specials = [word2idx.get(t) for t in ('<PAD>', '<SOS>', '<EOS>')]
        if specials != [0, 1, 2]:
            reason = 'The special tokens should have the indices <PAD> 0, <SOS> 1 and <EOS> 2.'
        elif set(word2idx) == set(expected):
            reason = ('The words are not in sorted (alphabetical) order after the special tokens. A set\'s order '
                      'changes every time Python starts, so a saved model would not fit the next session.')
        else:
            reason = 'word2idx should have the three special tokens and every word of the sentences, once.'
        raise AssertionError(f'{reason} For the sentences {sample}, word2idx should be {expected}, but it is '
                             f'{dict(word2idx)}.')
    inverse = {i: w for w, i in expected.items()}
    assert dict(idx2word) == inverse, f'idx2word should be the inverse of word2idx, {inverse}, but it is {dict(idx2word)}.'


def check_embedding_shapes(output_embedding_fr, output_embedding_en, src_batch, tgt_batch, embedding_size):
    """0 points: the shapes of the two embedding outputs."""
    errors = []
    for name, out, batch in (('French', output_embedding_fr, src_batch), ('English', output_embedding_en, tgt_batch)):
        want = (batch.size(0), batch.size(1), embedding_size)
        if tuple(out.shape) != want:
            errors.append(f'the {name} embedding output has shape {tuple(out.shape)}, expected {want}')
    assert not errors, 'Embedding shapes: ' + '; '.join(errors) + '.'


def check_positional_encoding(PositionalEncoding):
    pe = PositionalEncoding(embed_size=128, max_len=512)
    x = torch.zeros(2, 10, 128)
    with _quiet():
        out = pe(x)
    assert tuple(out.shape) == (2, 10, 128), (
        f'PositionalEncoding(128) on input of shape (2, 10, 128) should give the same shape, but it gave {tuple(out.shape)}.')
    buffers = dict(pe.named_buffers())
    assert 'pos_encoding' in buffers, 'The positional encodings should be registered as the buffer pos_encoding.'
    assert not any(p.requires_grad for p in pe.parameters()), 'The positional encoding should have no trainable parameters.'
    position = torch.arange(10, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, 128, 2).float() * (-math.log(10000.0) / 128))
    want = torch.zeros(10, 128)
    want[:, 0::2] = torch.sin(position * div_term)
    want[:, 1::2] = torch.cos(position * div_term)
    got = out[0].float()
    if not torch.allclose(got, want, atol=1e-4):
        diff = (got - want).abs()
        pos, dim = divmod(int(diff.argmax()), 128)
        hint = ''
        if torch.allclose(got[:, 0::2], want[:, 1::2], atol=1e-4) and torch.allclose(got[:, 1::2], want[:, 0::2], atol=1e-4):
            hint = ' Sine and cosine are swapped: sine goes to the even indices, cosine to the odd ones.'
        raise AssertionError((hint.strip() + ' ' if hint else 'The values are not the sinusoidal encoding '
                              '(sin(pos * div_term) at the even indices, cos at the odd ones). ') +
                             f'With zero embeddings the output should be the encoding itself; at position {pos}, '
                             f'index {dim}, it is {got[pos, dim].item():.4f}, expected {want[pos, dim].item():.4f}.')


# ---------------------------------------------------------------------------------------------------------------------
# Visible tests of Task 2

def _count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _logits(result):
    """MySimpleTransformer returns the logits; MyTransformer returns (decoder output, logits)."""
    return result[1] if isinstance(result, (tuple, list)) else result


def check_transformer(cls, PositionalEncoding, separate=False):
    """The model of Task 2.1 (separate=False) or Task 2.2 (separate=True), with vocabularies of 336 French and 201
    English words, on a batch of two random sentences (no data needed): the output shape, the number of parameters,
    the positional encoding, the causal mask, the memory padding mask (the decoder must not attend to the padded source
    positions), and (Task 2.2) encode() and decode()."""
    name = cls.__name__
    torch.manual_seed(0)
    vs, vt = 336, 201
    model = cls(vs, vt, 512, 2, 128, 3, 3, max_len=512)
    model.eval()
    g = torch.Generator().manual_seed(0)
    src = torch.randint(3, vs, (2, 8), generator=g)
    src[1, 6:] = PAD                                    # the second sentence is shorter
    tgt = torch.randint(3, vt, (2, 10), generator=g)
    tgt[:, 0] = SOS
    errors = []
    pe_calls = []
    hooks = [m.register_forward_hook(lambda *a: pe_calls.append(1)) for m in model.modules()
             if isinstance(m, PositionalEncoding)]

    def run(t, s=src):
        return model(s, t, src_padding_mask=model.create_pad_mask(s), tgt_padding_mask=model.create_pad_mask(t),
                     tgt_mask=model.get_tgt_mask(t))
    try:
        with _quiet():
            result = run(tgt)
    except RuntimeError as e:
        raise AssertionError(f'The forward pass of {name}(336, 201, 512, 2, 128, 3, 3) on a batch of 2 sentences failed '
                             f'(is batch_first=True set in nn.Transformer?): {str(e)[:300]}') from None
    for h in hooks:
        h.remove()
    if separate and not (isinstance(result, (tuple, list)) and len(result) == 2):
        raise AssertionError(f'{name}.forward() should return two tensors: the decoder output and the output of the '
                             f'final linear layer.')
    out = _logits(result)
    if tuple(out.shape) != (2, 10, vt):
        errors.append(f'For 2 target sentences of 10 tokens, the output should have shape (2, 10, {vt}) (batch, '
                      f'sequence, English vocabulary), but it has {tuple(out.shape)}.')
    n = _count(model)
    if n != 10641353:
        errors.append(f'{name}(336, 201, 512, 2, 128, 3, 3) should have 10641353 trainable parameters (two '
                      f'embeddings, nn.Transformer and one nn.Linear), but it has {n}.')
    if not hooks:
        errors.append(f'{name} has no PositionalEncoding layer.')
    elif not pe_calls:
        errors.append(f'{name}.forward() does not use its PositionalEncoding layer.')
    if not errors:
        changed = tgt.clone()
        changed[:, 6:] = torch.randint(3, vt, (2, 4), generator=g)
        with _quiet():
            out2 = _logits(run(changed))
        if not torch.allclose(out[:, :6], out2[:, :6], atol=1e-4):
            errors.append('The causal mask is not used: the output at the first 6 positions changed when only later '
                          'target tokens changed. The forward pass must give tgt_mask to the Transformer.')
    if not errors:
        padded = torch.cat([src, torch.full((2, 4), PAD, dtype=src.dtype)], dim=1)    # the same sentences, 4 more PAD
        with _quiet():
            out3 = _logits(run(tgt, padded))
        if not torch.allclose(out, out3, atol=1e-4):
            errors.append('The decoder attends to the padded source positions: the output changed when only the padding '
                          'at the end of the source changed. Give the source padding mask to the decoder too, as '
                          'memory_key_padding_mask' + (' (decode() gets it as memory_padding_mask from forward()).'
                                                       if separate else '.'))
    if separate and not errors:
        with _quiet():
            memory = model.encode(src, model.create_pad_mask(src))
            decoded = model.decode(tgt, memory, model.get_tgt_mask(tgt), model.create_pad_mask(tgt),
                                   model.create_pad_mask(src))
        if tuple(memory.shape) != (2, 8, 512):
            errors.append(f'encode() of 2 source sentences of 8 tokens should give shape (2, 8, 512), but gave '
                          f'{tuple(memory.shape)}.')
        elif tuple(decoded.shape) != (2, 10, 512):
            errors.append(f'decode() of 2 target sentences of 10 tokens should give shape (2, 10, 512), but gave '
                          f'{tuple(decoded.shape)}.')
        elif not torch.allclose(decoded, result[0], atol=1e-4):
            errors.append('forward() should pass the source through encode() and the target through decode(): its '
                          'decoder output differs from decode(tgt, encode(src, ...), ...).')
    assert not errors, ' '.join(errors)


# ---------------------------------------------------------------------------------------------------------------------
# Task 3: the saved model

def validation_data(tokenize_fr, tokenize_en, fr_word2idx, en_word2idx, seq_len=11, subset=SUBSET):
    """The first `subset` validation sentences of Task 3, in the order of val_loader, encoded as TranslationDataset
    encodes them (SOS, the words, EOS, cut to seq_len, padded with PAD). The validation sentences are the 20 % that
    random_split(..., generator=torch.Generator().manual_seed(42)) keeps for validation."""
    n = len(tokenize_fr)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    idx = perm[int(0.8 * n):][:subset]

    def encode(tokens, vocab):
        ids = ([vocab['<SOS>']] + [vocab.get(t, vocab['<PAD>']) for t in tokens] + [vocab['<EOS>']])[:seq_len]
        return ids + [vocab['<PAD>']] * (seq_len - len(ids))
    src = torch.tensor([encode(tokenize_fr[i], fr_word2idx) for i in idx])
    tgt = torch.tensor([encode(tokenize_en[i], en_word2idx) for i in idx])
    return src, tgt


def load_model(MyTransformer, data, vsize_src, vsize_tgt, device):
    """Your MyTransformer with the settings of Task 3, with the weights of data (the bytes of model.pth) loaded."""
    try:
        state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    except Exception as e:
        raise AssertionError(f'model.pth could not be read with torch.load ({type(e).__name__}: {e}).'[:250]) from None
    if isinstance(state, nn.Module):
        state = state.state_dict()
    model = MyTransformer(vsize_src, vsize_tgt, EMBED_SIZE, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_LAYERS,
                          max_len=MAX_LEN).to(device)
    try:
        model.load_state_dict(state)
    except Exception as e:
        size = re.search(r'size mismatch for ([\w.]+): copying a param with shape torch.Size\((\[[^\]]*\])\) from '
                         r'checkpoint, the shape in current model is torch.Size\((\[[^\]]*\])\)', str(e))
        lines = [l.strip() for l in str(e).split('\n') if l.strip()]
        reason = (f'{size.group(1)} has shape {size.group(2)} in the file but {size.group(3)} in the model' if size else
                  lines[1] if len(lines) > 1 else lines[0] if lines else type(e).__name__)
        raise AssertionError(f"model.pth does not fit your MyTransformer with the settings of Task 3 ({vsize_src} "
                             f"French and {vsize_tgt} English words): {reason}"[:300]) from None
    return model


def validation_loss(model, src, tgt, device, batch_size=256):
    """The validation loop of Task 3: batches of 256 (the last, partial one left out), target input tgt[:, :-1],
    expected output tgt[:, 1:], the padding masks and the causal mask, cross-entropy over every position; the mean
    over the batches, rounded to 4 decimals."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total, n = 0.0, len(src) // batch_size
    with _quiet():
        for b in range(n):
            s = src[b * batch_size:(b + 1) * batch_size].to(device)
            t = tgt[b * batch_size:(b + 1) * batch_size].to(device)
            t_in, t_out = t[:, :-1], t[:, 1:]
            result = model(s, t_in, src_padding_mask=model.create_pad_mask(s),
                           tgt_padding_mask=model.create_pad_mask(t_in).float(), tgt_mask=model.get_tgt_mask(t_in))
            out = _logits(result)
            total += criterion(out.reshape(-1, out.size(-1)), t_out.reshape(-1)).item()
    return round(total / n, 4)


def vocab_hint(fr_word2idx, en_word2idx):
    for name, vocab in (('fr_word2idx', fr_word2idx), ('en_word2idx', en_word2idx)):
        words = [w for w, _ in sorted(vocab.items(), key=lambda kv: kv[1])][3:]
        if words != sorted(words):
            return (f'Your {name} is not sorted: a set\'s order changes every time Python starts, so it is not the '
                    f'order you trained with.')
    return ''


def loss_message(loss, fr_word2idx, en_word2idx, advice=''):
    """Why the loss of model.pth is too high: the vocabulary hint first, if there is one."""
    hint = vocab_hint(fr_word2idx, en_word2idx)
    text = (f'The validation loss of model.pth on the first {SUBSET} validation sentences is {loss}; at most '
            f'{LOSS_LIMIT} is needed.')
    return f'{hint} {text}' if hint else f'{text} {advice}'.strip()


def saved_model_loss(MyTransformer, tokenized_fr, tokenized_en, fr_word2idx, en_word2idx, device):
    """The validation loss of the saved model.pth on the first 2048 validation sentences, as grading computes it."""
    data = submitted_model()
    assert data is not None, 'model.pth was not found next to your notebook. Submit it together with ex5.ipynb.'
    try:
        model = load_model(MyTransformer, data, len(fr_word2idx), len(en_word2idx), device)
    except AssertionError as e:
        hint = vocab_hint(fr_word2idx, en_word2idx)
        raise AssertionError(f'{hint} {e}' if hint else str(e)) from None
    src, tgt = validation_data(tokenized_fr, tokenized_en, fr_word2idx, en_word2idx)
    return validation_loss(model, src, tgt, device)


def check_saved_model(MyTransformer, tokenized_fr, tokenized_en, fr_word2idx, en_word2idx, device):
    loss = saved_model_loss(MyTransformer, tokenized_fr, tokenized_en, fr_word2idx, en_word2idx, device)
    assert loss <= LOSS_LIMIT, loss_message(loss, fr_word2idx, en_word2idx)
    print(f'The validation loss of your saved model.pth (on the first {SUBSET} validation sentences) is {loss}: at '
          f'most {LOSS_LIMIT} is needed.')


def load_trained_model(model, device, grading):
    """Load model.pth (during grading: the submitted copy) into model. Returns a short message; never raises, so that
    Task 4 can still run."""
    data = submitted_model() if grading else (open('model.pth', 'rb').read() if os.path.isfile('model.pth') else None)
    if data is None:
        return 'model.pth was not found: Task 4 will translate with an untrained model.'
    try:
        state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        model.load_state_dict(state.state_dict() if isinstance(state, nn.Module) else state)
    except Exception as e:
        return f'model.pth could not be loaded into model ({type(e).__name__}: {str(e)[:200]}); Task 4 will translate with an untrained model.'
    return 'model.pth loaded into model.'
