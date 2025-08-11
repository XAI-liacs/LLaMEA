import ast
import re
from difflib import SequenceMatcher
from typing import List, Tuple
import numpy as np
import subprocess
import os


class NoCodeException(Exception):
    """Could not extract generated code."""

    pass


def handle_timeout(signum, frame):
    """Raise a timeout exception"""
    raise TimeoutError


_HUNK_HDR = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*$")


def _norm(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    if not s.endswith("\n"):
        s += "\n"
    return s


def _split_lines(s: str) -> List[str]:
    # keepends=True so we preserve exact text
    return s.splitlines(keepends=True)


def _parse_hunks(diff: str) -> List[List[str]]:
    """Return a list of hunk payloads (each is list of lines including markers)."""
    diff = _norm(diff)
    lines = diff.splitlines(keepends=True)

    hunks: List[List[str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@ "):
            # capture header
            m = _HUNK_HDR.match(line[0:-1] if line.endswith("\n") else line)
            if not m:
                # tolerant: still treat as header
                pass
            i += 1
            payload: List[str] = [line]  # include header at index 0
            # collect payload lines that *look* like unified diff content
            while i < len(lines):
                l = lines[i]
                if (
                    l.startswith("@@ ")
                    or l.startswith("--- ")
                    or l.startswith("diff --git ")
                    or l.startswith("Index: ")
                ):
                    break
                # if a payload line is unmarked, treat it as context
                if l and l[0] not in (" ", "+", "-", "\\"):
                    l = " " + l
                payload.append(l)
                i += 1
            hunks.append(payload)
        else:
            i += 1
    return hunks


def _old_new_from_hunk(payload: List[str]) -> Tuple[List[str], List[str]]:
    """Given a hunk payload (header + lines), produce old_seq and new_seq (lists of raw text lines)."""
    old_seq: List[str] = []
    new_seq: List[str] = []
    # skip header at payload[0]
    for l in payload[1:]:
        if not l:  # shouldn't happen
            continue
        tag = l[0]
        body = l[1:]
        if tag == " ":
            old_seq.append(body)
            new_seq.append(body)
        elif tag == "-":
            old_seq.append(body)
        elif tag == "+":
            new_seq.append(body)
        elif tag == "\\":  # "\ No newline at end of file" – ignore for content
            pass
        else:
            # unmarked -> treat as context
            old_seq.append(l)
            new_seq.append(l)
    return old_seq, new_seq


def _leading_trailing_context_mask(
    old_pairs: List[Tuple[str, bool]]
) -> Tuple[int, int]:
    """Return counts of leading and trailing context-only lines in old_pairs."""
    lead = 0
    for t, required in old_pairs:
        if required:
            break
        lead += 1
    trail = 0
    for t, required in reversed(old_pairs):
        if required:
            break
        trail += 1
    return lead, trail


def _find_with_fuzz(
    text_lines: List[str], old_seq: List[str], ctx_flags: List[bool], max_fuzz: int
) -> int:
    """
    Find the start index where old_seq occurs in text_lines.
    ctx_flags[i] == False for context lines (originating from ' '), True for required lines (originating from '-').
    Fuzz allows dropping up to max_fuzz context lines from the *start and/or end* when locating the block.
    Returns the *start index for the full, non-dropped old_seq* slice if found, else -1.
    """
    n = len(old_seq)
    if n == 0:
        return 0

    # exact match first
    for i in range(0, len(text_lines) - n + 1):
        if text_lines[i : i + n] == old_seq:
            return i

    # prepare to drop context lines at the edges only
    lead_ctx, trail_ctx = _leading_trailing_context_mask(list(zip(old_seq, ctx_flags)))
    max_lead_drop = min(max_fuzz, lead_ctx)
    max_trail_drop = min(max_fuzz, trail_ctx)

    for d_lead in range(0, max_lead_drop + 1):
        for d_trail in range(0, max_trail_drop + 1):
            if d_lead == 0 and d_trail == 0:
                continue
            sub = old_seq[d_lead : n - d_trail]
            if not sub:
                continue
            m = len(sub)
            for i in range(0, len(text_lines) - m + 1):
                if text_lines[i : i + m] == sub:
                    start = i - d_lead
                    if start < 0:
                        continue
                    end = start + n
                    if end > len(text_lines):
                        continue
                    # we accept mismatches only on the dropped context edges;
                    # the interior must match exactly
                    if text_lines[start + d_lead : end - d_trail] == sub:
                        return start
    return -1


def apply_unified_diff(text: str, diff: str, max_fuzz: int = 5) -> str:
    """
    Pure-Python unified-diff applier.
    - Ignores file headers; applies hunks by content.
    - Fuzz: allows dropping up to `max_fuzz` *leading/trailing context* lines to find a match.
    - Raises ValueError if a hunk can't be placed.
    """
    text = _norm(text)
    lines = _split_lines(text)

    hunks = _parse_hunks(diff)
    if not hunks:
        raise ValueError("No @@ hunks found in diff.")

    for idx, payload in enumerate(hunks, 1):
        old_seq, new_seq = _old_new_from_hunk(payload)
        # mark which old_seq lines were required (came from '-') vs context
        ctx_flags: List[bool] = []
        for l in payload[1:]:
            if not l:
                continue
            tag = l[0]
            if tag == " ":
                ctx_flags.append(False)
            elif tag == "-":
                ctx_flags.append(True)
            elif tag == "+":
                # additions do not appear in old_seq; skip for ctx_flags
                continue
            elif tag == "\\":
                continue
            else:
                ctx_flags.append(False)
        if len(ctx_flags) != len(old_seq):
            # Should not happen; make all required to be safe
            ctx_flags = [True] * len(old_seq)

        pos = _find_with_fuzz(lines, old_seq, ctx_flags, max_fuzz=max_fuzz)
        if pos < 0:
            # compact debug context
            want_preview = "".join(old_seq[:5])
            print(
                f"Could not place hunk #{idx} (len old={len(old_seq)} new={len(new_seq)}). "
                f"Try increasing max_fuzz. Preview of expected start:\n{want_preview}"
            )
            return text

        # apply replacement
        lines[pos : pos + len(old_seq)] = new_seq

    return "".join(lines)


def discrete_power_law_distribution(n, beta):
    """
    Power law distribution function from:
    # Benjamin Doerr, Huu Phuoc Le, Régis Makhmara, and Ta Duy Nguyen. 2017.
    # Fast genetic algorithms.
    # In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17).
    # Association for Computing Machinery, New York, NY, USA, 777–784.
    # https://doi.org/10.1145/3071178.3071301
    """

    def discrete_power_law(n, alpha, beta):
        half_n = int(n / 2)
        C_beta_half_n = 0
        for i in range(1, half_n + 1):
            C_beta_half_n += i ** (-beta)
        probability_alpha = C_beta_half_n ** (-1) * alpha ** (-beta)
        return probability_alpha

    half_n = int(n / 2)
    elements = [alpha for alpha in range(1, half_n + 1)]
    probabilities = [discrete_power_law(n, alpha, beta) for alpha in elements]
    if elements == []:
        return 0.05
    else:
        sample = np.random.choice(elements, p=probabilities)
        return sample / n


def code_distance(a, b):
    """Return a rough distance between two solutions based on their ASTs.

    The function accepts either :class:`Solution` objects or raw code strings
    and computes ``1 - similarity`` of their abstract syntax trees using
    :class:`difflib.SequenceMatcher` on the dumped AST representations.
    ``1.0`` is returned on parsing errors or when the inputs cannot be
    processed.

    Args:
        a: The first solution or Python source code.
        b: The second solution or Python source code.

    Returns:
        float: A value in ``[0, 1]`` indicating dissimilarity of the code.
    """

    code_a = getattr(a, "code", a)
    code_b = getattr(b, "code", b)
    try:
        tree_a = ast.parse(code_a)
        tree_b = ast.parse(code_b)
        return 1 - SequenceMatcher(None, ast.dump(tree_a), ast.dump(tree_b)).ratio()
    except Exception:
        return 1.0
