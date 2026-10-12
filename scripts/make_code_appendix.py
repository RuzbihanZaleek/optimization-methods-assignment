"""Create the plain-text code appendix required by the assignment brief.

Owner: U.D.L.L De Silva (Member 2)

It copies every Python source file and the code cells of every notebook into one UTF-8 text
file, with a header per file, so the appendix always matches the code in the repository.
The Git commit it was generated from is written at the top.

Run from the repository root (after the final commit):
    python scripts/make_code_appendix.py
"""

import json
import subprocess
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "report" / "code_appendix.txt"
RULE = "=" * 100


def source_files():
    """All files that go into the appendix, in a fixed order."""
    python_files = sorted((REPO_ROOT / "src").glob("*.py")) + sorted((REPO_ROOT / "scripts").glob("*.py"))
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    return python_files, notebooks


def notebook_code(path):
    """Only the code cells of a notebook (outputs and text cells are left out)."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    parts = []
    for number, cell in enumerate(cells, start=1):
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        parts.append(f"# ---- code cell {number} ----\n{source.rstrip()}\n")
    return "\n".join(parts)


def current_commit():
    """Short hash of the current Git commit, so the appendix can be traced to the repository."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    python_files, notebooks = source_files()
    sections = [(path, path.read_text(encoding="utf-8")) for path in python_files]
    sections += [(path, notebook_code(path)) for path in notebooks]

    header = [
        RULE,
        "CODE APPENDIX - Delivery Optimization (Capacitated Vehicle Routing Problem)",
        "IT50802 Optimization Methods, MSc in Artificial Intelligence, SLIIT",
        "M.Z.M. Ruzbihan (MS26911458) and U.D.L.L De Silva (MS26903224)",
        f"Generated on {date.today().isoformat()} from Git commit {current_commit()}",
        "Repository: https://github.com/RuzbihanZaleek/optimization-methods-assignment",
        RULE,
        "",
        "Contents",
    ]
    for number, (path, text) in enumerate(sections, start=1):
        relative = path.relative_to(REPO_ROOT).as_posix()
        kind = " (code cells only)" if path.suffix == ".ipynb" else ""
        header.append(f"  {number:2d}. {relative}{kind} - {len(text.splitlines())} lines")
    header.append("")

    body = []
    for number, (path, text) in enumerate(sections, start=1):
        relative = path.relative_to(REPO_ROOT).as_posix()
        body += [RULE, f"{number}. {relative}", RULE, text.rstrip() if text.strip() else "(empty file)", ""]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    total_lines = sum(len(text.splitlines()) for _, text in sections)
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT).as_posix()}: {len(sections)} files, {total_lines} lines of code")


if __name__ == "__main__":
    main()
