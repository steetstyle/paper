"""Sentence-level text checker: AI-authorship risk + plagiarism, per sentence.

The package is deliberately layered so every signal is optional and the tool
still produces a useful report when it degrades:

``checker_app.domain``
    Pure data: enums, dataclasses, tokenisation, sentence segmentation with
    character-exact locations. No I/O, no ML, no framework imports.
``checker_app.services``
    One module per signal (stylometry, perplexity, classifier, plagiarism,
    code/AST) plus scoring, sources, reporting and the orchestrating runner.
``checker_app.cli``
    A thin Typer + Rich front end.
"""

from __future__ import annotations

__version__ = "0.1.0"
