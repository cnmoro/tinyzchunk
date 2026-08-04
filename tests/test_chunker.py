"""Behavioural tests for tinyzchunk.

These assert the properties a chunker must never violate, regardless of how the
models are retrained: chunks are exact substrings, no content is dropped, no cut
lands inside a word, size limits hold, and equivalent inputs (CRLF, unicode
punctuation) produce equivalent output.

    pytest tests/
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyzchunk.chunker import Chunker, _protected_spans          # noqa: E402
from tinyzchunk.features import (extract_features, n_features,    # noqa: E402
                                 normalize_text, normalize_with_map)
from tinyzchunk.labels import snap_positions                      # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "tinyzchunk", "weights.npz")
needs_weights = pytest.mark.skipif(
    not os.path.exists(WEIGHTS), reason="weights not present in the source tree")


@pytest.fixture(scope="module")
def chunker():
    return Chunker()


PROSE = (
    "Introduction\n\n"
    "The system processes documents in several stages. Each stage is "
    "independent and can be replaced without touching the others.\n\n"
    "Configuration\n\n"
    "Settings are read from a file at startup. Defaults are conservative so "
    "that a fresh install behaves sensibly.\n"
)

QA = (
    "What is the deadline for registration?\nRegistration closes on 30 September.\n"
    "Who can I contact for help?\nWrite to the support desk at any time.\n"
    "Is there a fee?\nNo, participation is free of charge for everyone.\n"
)

MARKDOWN = (
    "## Install\n\nRun the installer and follow the prompts.\n\n"
    "```python\ndef main():\n    return 0\n\nif True:\n    main()\n```\n\n"
    "## Options\n\n| flag | meaning |\n|---|---|\n| -v | verbose |\n| -q | quiet |\n"
)


# ------------------------------------------------------------------ pure units

def test_normalization_is_one_to_one_except_cr():
    for s in ["plain", "a b", "“quoted”", "em—dash", "ellipsis…", "\x0cff"]:
        assert len(normalize_text(s)) == len(s), repr(s)


def test_crlf_index_map_points_back_at_the_original():
    raw = "alpha\r\nbeta\r\ngamma"
    norm, index_map = normalize_with_map(raw)
    assert norm == "alpha\nbeta\ngamma"
    assert len(index_map) == len(norm)
    for i, ch in enumerate(norm):
        assert raw[index_map[i]] in (ch, "\r")


def test_features_have_a_stable_width_and_are_finite():
    for s in ["", "a", "a\nb", PROSE, MARKDOWN, "x" * 3000]:
        X = extract_features(s)
        assert X.shape == (len(normalize_text(s)), n_features())
        assert np.isfinite(X).all()


def test_snap_positions_never_leaves_a_boundary_inside_a_word():
    text = "One day something happened. Another thing followed."
    for p in snap_positions(text, [2, 5, 9, 30]):
        assert not (text[p - 1].isalnum() and text[p].isalnum())


def test_protected_spans_cover_fences_and_tables():
    spans = _protected_spans(MARKDOWN)
    assert spans, "expected a fenced block and a table to be protected"
    fence = MARKDOWN.index("```")
    assert any(a <= fence < b for a, b in spans)


# ---------------------------------------------------------------- chunker core

@needs_weights
@pytest.mark.parametrize("text", [PROSE, QA, MARKDOWN])
def test_chunks_are_exact_substrings(chunker, text):
    for ch in chunker.chunk(text):
        assert ch in text


@needs_weights
@pytest.mark.parametrize("text", [PROSE, QA, MARKDOWN])
def test_no_content_is_dropped(chunker, text):
    joined = "".join("".join(chunker.chunk(text)).split())
    assert joined == "".join(text.split())


@needs_weights
@pytest.mark.parametrize("text", [PROSE, QA, MARKDOWN])
def test_no_chunk_starts_inside_a_word(chunker, text):
    for ch in chunker.chunk(text):
        i = text.find(ch)
        assert not (i > 0 and text[i - 1].isalnum() and text[i].isalnum())


@needs_weights
def test_max_chunk_chars_is_respected(chunker):
    text = ("Sentence number one here. Sentence number two here. "
            "Sentence number three here. ") * 90
    c = Chunker(max_chunk_chars=600, min_chunk_chars=50)
    assert all(len(ch) <= 600 for ch in c.chunk(text))


@needs_weights
def test_crlf_matches_lf(chunker):
    for text in (PROSE, QA, MARKDOWN):
        lf = chunker.chunk(text)
        crlf = [c.replace("\r\n", "\n") for c in chunker.chunk(text.replace("\n", "\r\n"))]
        assert crlf == lf


@needs_weights
def test_short_single_line_document_is_left_whole(chunker):
    text = "This document has no line structure at all. " * 6
    assert len(chunker.chunk(text.strip())) == 1


@needs_weights
def test_code_fence_is_never_split(chunker):
    chunks = chunker.chunk(MARKDOWN)
    body = "def main():\n    return 0"
    assert any(body in ch for ch in chunks), "the fenced block was cut apart"


@needs_weights
@pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "a", "\r\n\r\n"])
def test_degenerate_inputs_do_not_raise(chunker, text):
    out = chunker.chunk(text)
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)


@needs_weights
def test_very_long_document_stays_bounded(chunker):
    text = (PROSE + "\n") * 400
    chunks = chunker.chunk(text)
    assert chunks
    assert max(len(c) for c in chunks) <= chunker.max_chunk_chars


@needs_weights
def test_weights_schema_mismatch_is_rejected(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, W1=np.zeros((4, 7), dtype=np.float32),
                        b1=np.zeros(4, dtype=np.float32),
                        SCHEMA=np.array("deadbeefcafe"))
    with pytest.raises(ValueError, match="feature schema"):
        Chunker(weights_path=str(bad))
