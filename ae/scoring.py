#!/usr/bin/env python3
"""Scoring helpers for the dLLM serving benchmark evaluation sets.

Public API
----------
score(task, prediction, reference, extra) -> bool | None

    gsm8k / mgsm : numeric exact match.  The reference answer and the model
                   prediction are both reduced to a single number using the
                   same extraction logic as the artifact harness
                   (artifact-public/ae/quality.py: `#### <n>` marker first,
                   otherwise the last number in the text).
    humaneval    : exact match between the reference solution and the code
                   block extracted from the prediction.  NOTE: this is the
                   plain exact-match baseline requested for the data-prep
                   stage; the real HumanEval metric is pass@1 from executing
                   `extra['test']` against the candidate.
    mt_bench     : always None (model-judged, not scored here).

The numeric extraction here is a verbatim port of `answer()` in
artifact-public/ae/quality.py.  That file cannot be imported directly (it
imports aiohttp and the `run` module at top level), so the logic is mirrored
rather than reused; tests/test_scoring.py pins the two to the same results.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
import textwrap
from typing import Optional

__all__ = [
    "score",
    "extract_number",
    "normalize_number",
    "final_answer_number",
    "extract_code",
    "normalize_code",
]

# Identical to quality.py: answer()
_NUMERIC_MARKER = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_NUMERIC_ANY = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_FENCE = re.compile(r"```(?:python3?|py)?[^\S\n]*\n(.*?)```", re.DOTALL)


def normalize_number(token: Optional[str]) -> Optional[str]:
    """Canonical decimal string, e.g. '18.0' -> '18', '1,234' -> '1234'."""
    if not token:
        return None
    try:
        value = Decimal(token.replace(",", "").strip()).normalize()
    except (InvalidOperation, AttributeError, ValueError):
        return None
    # quality.py gets str(Decimal.normalize()), which renders 18 as '1.8E+1'.
    # Both sides of a comparison pass through here, so the exponent form is
    # harmless for correctness, but plain fixed-point is easier to read in logs.
    if value == value.to_integral_value():
        return str(value.to_integral_value())
    return format(value.normalize(), "f")


def extract_number(text: Optional[str]) -> Optional[str]:
    """Reduce free-form text to one canonical number, quality.py semantics."""
    if not text:
        return None
    match = _NUMERIC_MARKER.search(text)
    if match is None:
        matches = _NUMERIC_ANY.findall(text)
        token = matches[-1] if matches else None
    else:
        token = match.group(1)
    return normalize_number(token) if token else None


def final_answer_number(text: Optional[str]) -> Optional[str]:
    """Convert a reference solution to the bare numeric `answer` field.

    Used at dataset-build time so MGSM holds only the final number, matching
    the on-disk GSM8K convention.  Falls back to the last number in the text
    when no `####` marker is present.
    """
    return extract_number(text)


def extract_code(prediction: Optional[str], extra: Optional[dict] = None) -> Optional[str]:
    """Pull a code block out of a model response.

    Preference order: fenced block (last one, models often restate), then the
    whole text when it already looks like code, then a raw text span carrying
    the entry point signature.

    Note that HumanEval `canonical_solution` values are indented function
    *bodies* -- the signature lives in `prompt` -- so an indented first line
    is a normal code shape here, not prose. The final fallback therefore
    returns the text as-is: leading indentation is load-bearing and is removed
    later by `normalize_code`.
    """
    if not prediction:
        return None
    fences = _FENCE.findall(prediction)
    if fences:
        # Keep indentation: a fenced body-only answer still needs dedenting.
        return fences[-1].strip("\n") or None
    # Column-0 code is a complete program. An indented first line is a body
    # (HumanEval canonical_solution shape) and must keep its indentation for
    # `normalize_code`, so test the unstripped text here.
    if prediction.lstrip("\n").startswith(("def ", "from ", "import ", "class ")):
        return prediction.lstrip("\n").rstrip()
    extra = extra or {}
    entry = extra.get("entry_point")
    if entry:
        match = re.search(rf"def\s+{re.escape(entry)}\s*\(", prediction)
        if match:
            return prediction[match.start():]
    return prediction if prediction.strip() else None


def normalize_code(code: Optional[str]) -> Optional[str]:
    """Put a code block in comparable form.

    Drops surrounding blank lines, removes the common indentation, and strips
    trailing whitespace per line.

    Order matters: canonical solutions are *bodies* nested one level under the
    prompt's signature, so `textwrap.dedent` must see the original leading
    whitespace. Stripping first would leave the body's inner lines indented by
    the outer level and break the comparison. `textwrap.dedent` passes lines
    that are pure whitespace through untouched; the final `rstrip` on each
    line reduces those to empty.
    """
    if not code:
        return None
    lines = textwrap.dedent(code).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines) or None


def _program_form(prediction: Optional[str]) -> Optional[str]:
    """The prediction normalized as-is, when it is already column-0 code.

    Some HumanEval prompts carry a preamble before the signature (HumanEval/64
    starts with a `FIX = ...` block), so re-anchoring at the entry point would
    discard legitimate code. Leading blank lines are ignored; leading
    indentation means the text is a function body, not a program.
    """
    if not prediction or not prediction.strip():
        return None
    if prediction.lstrip("\n")[:1] in (" ", "\t"):
        return None
    return normalize_code(prediction)


def score(
    task: str,
    prediction: Optional[str],
    reference: Optional[str],
    extra: Optional[dict] = None,
) -> Optional[bool]:
    """Return whether `prediction` matches `reference` for `task`.

    None means "not auto-scorable" (mt_bench) rather than "incorrect".

    For humaneval the reference is `canonical_solution`, a function *body*:
    the signature lives in the record's `question`. A model normally answers
    with a whole program, so the prediction is accepted if it matches either
    form -- the body itself, or prompt + body, which is the full program.
    """
    if task in ("gsm8k", "mgsm"):
        pred = extract_number(prediction)
        ref = extract_number(reference)
        return pred is not None and ref is not None and pred == ref
    if task == "humaneval":
        ref = normalize_code(reference)
        if ref is None:
            return False
        candidates = {
            c for c in (normalize_code(extract_code(prediction, extra)),
                        _program_form(prediction))
            if c is not None
        }
        if not candidates:
            return False
        if ref in candidates:
            return True
        prompt = (extra or {}).get("question")
        if prompt:
            full = normalize_code(f"{prompt.rstrip()}\n{reference}")
            return full is not None and full in candidates
        return False
    if task == "mt_bench":
        return None
    raise ValueError(f"unknown task: {task!r}")


def _demo() -> int:
    """Self-check used as a smoke test; no network, no files."""
    cases = [
        ("gsm8k", "The total is 16 - 3 - 4 = 9.\n#### 18", "#### 18", True),
        ("gsm8k", "I think the answer is 18.", "#### 18", True),
        ("gsm8k", "answer: 17", "#### 18", False),
        ("mgsm", "#### 70,000", "70000", True),
        ("mgsm", "no numbers here", "#### 3", False),
        ("mgsm", "#### 1.50", "1.5", True),
    ]
    ok = True
    for task, pred, ref, want in cases:
        got = score(task, pred, ref, None)
        ok &= got == want
        print(f"{'ok ' if got == want else 'BAD'} {task}: pred={pred!r} ref={ref!r} -> {got}")
    ext = {"entry_point": "has_close_elements"}
    sol = "def f():\n    return 1"
    print("humaneval fenced:", score("humaneval", f"```python\n{sol}\n```", sol, ext))
    # HumanEval canonical_solution shape: an indented body, no signature.
    body = "    for x in y:\n        return x"
    got = score("humaneval", body, body, ext)
    ok &= got is True
    print(f"{'ok ' if got is True else 'BAD'} humaneval indented body -> {got}")
    got = score("humaneval", "    for x in y:\n        return z", body, ext)
    ok &= got is False
    print(f"{'ok ' if got is False else 'BAD'} humaneval body mismatch -> {got}")
    print("mt_bench:", score("mt_bench", "anything", "", {}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
