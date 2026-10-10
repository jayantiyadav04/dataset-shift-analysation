"""Check the supplied specification pack; this does not test an implemented app."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


lock = read("docs/00_OBJECTIVE_LOCK.md")
objective = lock.split("## Final objective\n\n", 1)[1].split("\n\n##", 1)[0].strip()
proposal = read("docs/01_PROJECT_PROPOSAL.md")
abstract = proposal.split("## Abstract\n\n", 1)[1].split("\n\nWord count:", 1)[0].strip()
assert len(abstract.split()) == 150, "Proposal abstract must remain 150 words"
assert objective in proposal, "Proposal objective differs from locked wording"
assert objective in read("prompts/MASTER_PROMPT.md"), "Master prompt objective differs"
prompts = sorted((ROOT / "prompts").glob("PHASE_*.md"))
assert [p.name for p in prompts] == [f"PHASE_{n:02d}.md" for n in range(1, 9)]
roadmap = read("docs/14_EIGHT_PHASE_ROADMAP.md")
assert len(re.findall(r"^## Phase [1-8] ", roadmap, re.M)) == 8
assert (9 + 1 + 12) * 3 * 3 * 5 == 990
assert (9 + 1 + 12) * 5 == 110
assert 3 * 3 * 5 * 4 == 180
# Inspect project Markdown, not installed third-party documentation in .venv.
markdown_paths = [*ROOT.glob("*.md")]
for directory in ("docs", "prompts", "templates", "reports"):
    markdown_paths.extend((ROOT / directory).rglob("*.md"))
for path in markdown_paths:
    text = path.read_text(encoding="utf-8")
    assert "\ue200genui" not in text and "【turn" not in text, f"Tool token in {path}"
    for ref in re.findall(r"\bdocs/([A-Za-z0-9_]+\.md)", text):
        if ref.startswith("phase_"):
            continue  # Planned build outputs, not existing documentation.
        assert (ROOT / "docs" / ref).is_file(), f"Missing referenced document: {ref}"
print(
    "PASS: objective wording, 150-word abstract, eight phases, run counts and document references"
)
print(
    "Scope: documentation consistency only. No project experiment has been executed by this check."
)
