"""Render `d3text.metric_docs.glossary` into a page at build time.

A page holds `<!-- metric-glossary: STAGE -->` where the table for `STAGE`
should appear, so the reference page cannot drift from what a run logs.
"""

import re

from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

import d3text.metric_docs as metric_docs

_PLACEHOLDER = re.compile(r"<!-- metric-glossary: (\w+) -->")


def on_page_markdown(
    markdown: str, page: Page, config: object, files: Files
) -> str:
    """Replace every glossary placeholder on the page with its table."""

    def render(match: re.Match[str]) -> str:
        stage = match.group(1)
        if stage not in metric_docs.STAGES:
            raise ValueError(
                f"{page.file.src_path}: no metric glossary for stage "
                f"{stage!r}; known stages: {sorted(metric_docs.STAGES)}"
            )
        return metric_docs.glossary(stage)

    return _PLACEHOLDER.sub(render, markdown)
