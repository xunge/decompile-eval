"""Official LLM4Decompile Edit Similarity (ES).

Reference implementation:
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target = '\\n'.join(target_lines)
    ... same for prediction ...
    ES = 1 - editdistance.eval(target, prediction) / max(len(target), len(prediction))

The official script uses the C ``editdistance`` package.  A pure-Python
Levenshtein fallback is provided so the API still works if the optional wheel
is unavailable (slower, but exact).
"""
from __future__ import annotations

from .db import connect

try:  # pragma: no cover - exercised through the fallback test
    import editdistance as _editdistance
except Exception:  # pragma: no cover
    _editdistance = None


def normalize_source(text: str | None) -> str:
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    return '\n'.join(lines)


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, ch in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ch != other),
            ))
        previous = current
    return previous[-1]


def edit_similarity(reference: str | None, prediction: str | None) -> float:
    """Return official Edit Similarity in [0, 1]."""
    target = normalize_source(reference)
    generated = normalize_source(prediction)
    longest = max(len(target), len(generated))
    if longest == 0:
        return 1.0
    if _editdistance is not None:
        distance = _editdistance.eval(target, generated)
    else:
        distance = _levenshtein(target, generated)
    return max(0.0, min(1.0, 1.0 - distance / longest))


def _normalized_prediction(source: str | None) -> str:
    # The official metric reads the generated .c file, not markdown fences.
    # Strip optional fences so a fenced answer is compared as source code.
    from .sandbox import extract_c
    return extract_c(source or '')


def backfill_edit_similarity() -> int:
    """Fill missing ES values for historical rows from stored outputs.

    Only the derived metric column is added; model outputs, compile status, RE,
    TCP and all historical decisions are left untouched.  Returns the number of
    rows updated.
    """
    with connect() as db:
        rows = [dict(row) for row in db.execute('''
            SELECT r.id,r.model_output,r.status,s.reference_source
            FROM results r JOIN samples s ON s.id=r.sample_id
            WHERE r.edit_similarity IS NULL
              AND r.status NOT IN ('INFERENCE_ERROR','INPUT_TOO_LONG',
                                   'GENERATED','COMPILING')
        ''')]
    updates = []
    for row in rows:
        updates.append((round(edit_similarity(
            row['reference_source'],
            _normalized_prediction(row['model_output'])), 6), row['id']))
    if not updates:
        return 0
    with connect() as db:
        db.executemany('UPDATE results SET edit_similarity=? WHERE id=?', updates)
    return len(updates)
