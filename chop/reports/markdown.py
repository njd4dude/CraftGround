"""Writing the generated part of a Markdown report.

A report keeps everything outside the AUTO-GENERATED markers (the Notes you write) and only
replaces what is between them."""

import re
from pathlib import Path


START = "<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->"

END = "<!-- AUTO-GENERATED:END -->"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def mean_ci(eps: list[int]) -> tuple[float, float]:
    """Mean and the half-width of its 95% range."""
    n = len(eps)
    mean = sum(eps) / n
    sd = (sum((x - mean) ** 2 for x in eps) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, 1.96 * sd / n ** 0.5


def write_report(path: Path, template: str, generated: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else template
    if START not in text or END not in text:
        text = template + "\n" + text  # markers removed by hand: put a fresh block on top
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    path.write_text(f"{before}{START}\n{generated}\n{END}{after}", encoding="utf-8")
    print(f"Updated {path}")


def page_template(world: str, title: str) -> str:
    """A report page: its title, a link back to the folder's README.md index, the generated block."""
    return f"""# {world}: {title}

[← {world} index](README.md)

{START}
{END}
"""


def body(section: str) -> str:
    """A section without its own "## Title" line (the page's title replaces it)."""
    return re.sub(r"\A## [^\n]*\n+", "", section)


def write_pages(folder: Path, world: str, index_template: str, stamp: str,
                pages: list[tuple[str, str, str]], sections: dict[str, str]) -> None:
    """Write one page per (file, title, what it holds) from sections[file], then the README.md index."""
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for name, title, about in pages:
        text = body(sections[name]) or "_No results yet._"
        write_report(folder / name, page_template(world, title), f"{stamp}\n\n{text}")
        index.append(f"- [{title}]({name}): {about}")
    write_report(folder / "README.md", index_template, f"{stamp}\n\n" + "\n".join(index))
