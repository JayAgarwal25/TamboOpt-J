"""Print only the executable tokens of a Python file.

Refactor guard: comments and docstrings are stripped, so two revisions of a file
that differ only in prose produce identical output. Snapshot the output before a
refactor, re-run after, and diff — any difference is a real change to executable
code.

Usage:
    python tools/codeonly.py path/to/file.py
"""

import sys
import tokenize

# A STRING token that directly follows one of these (or starts the file) is a
# docstring: nothing has begun a statement yet, so the string is a bare
# expression at the top of a module, class or function body.
_DOCSTRING_PRECEDERS = frozenset(
    (tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT, tokenize.NL)
)


def code_tokens(path):
    """Yield the executable token strings of the file at *path*."""
    with tokenize.open(path) as fh:
        prev_significant = None  # token type of the last non-trivia token
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and (
                prev_significant is None or prev_significant in _DOCSTRING_PRECEDERS
            ):
                # Docstring (or a stray bare string literal): not executable code.
                prev_significant = tok.type
                continue
            if tok.type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            prev_significant = tok.type
            if tok.string.strip():
                yield tok.string


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} FILE.py", file=sys.stderr)
        return 2
    for text in code_tokens(argv[1]):
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
