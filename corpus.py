"""Knowledge base loading and structure aware chunking.

Each document is Markdown with a small YAML front matter block. Chunks never
cross a ``##`` heading, because a retrieved chunk must make sense on its own
and because the heading tells the generator what kind of content it holds:

* ``actions``       imperative steps that may be quoted into a response plan
* ``prohibitions``  things that must not be done
* ``context``       explanation, used for summaries and question answering
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LEVEL = re.compile(r"\bLevel\s+(WATCH|INCIPIENT|GROWING|CRITICAL)\b", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:\*|\d+\.)\s+(.*)$")


@dataclass
class Document:
    doc_id: str
    title: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    kind: str  # actions | prohibitions | context
    level: str | None
    tags: list[str]
    zone_tags: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bullets(self) -> list[str]:
        out = []
        for line in self.text.splitlines():
            match = _BULLET.match(line)
            if match:
                out.append(match.group(1).strip())
        return out

    def search_text(self) -> str:
        """Text used for indexing: the heading path gives short chunks their context."""
        return f"{self.title}. {self.section}. {self.text}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "title": self.title, "section": self.section,
            "text": self.text, "kind": self.kind, "level": self.level, "tags": self.tags,
            "zone_tags": self.zone_tags, "metadata": self.metadata,
        }

    @staticmethod
    def from_payload(p: dict[str, Any]) -> Chunk:
        return Chunk(p["chunk_id"], p["doc_id"], p["title"], p["section"], p["text"], p["kind"], p.get("level"),
                     list(p.get("tags", [])), list(p.get("zone_tags", [])), dict(p.get("metadata", {})))


def parse_front_matter(front: str) -> dict[str, Any]:
    """Parse ``key: value`` lines. Values in square brackets become lists.

    A deliberately forgiving parser: knowledge base authors are safety
    officers, not YAML experts, and a title containing a colon must not
    break the index build.
    """
    meta: dict[str, Any] = {}
    for line in front.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key.strip()] = value.strip("\"'")
    return meta


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        _, front, body = raw.split("---", 2)
        meta = parse_front_matter(front)
    doc_id = str(meta.get("id") or path.stem.upper())
    title = str(meta.get("title") or path.stem.replace("_", " ").title())
    return Document(doc_id, title, body.strip(), meta, str(path))


def load_documents(directory: str | Path) -> list[Document]:
    paths = sorted(Path(directory).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No knowledge base documents found in {directory}")
    return [parse_document(p) for p in paths]


def section_kind(heading: str) -> str:
    h = heading.lower()
    if h.startswith("do not") or "never" in h or "prohibit" in h:
        return "prohibitions"
    if "action" in h or "procedure" in h or "steps" in h:
        return "actions"
    return "context"


def _pack(blocks: list[str], max_words: int, overlap_words: int) -> list[str]:
    """Greedily pack paragraphs or bullets into chunks of at most ``max_words``."""
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for block in blocks:
        words = len(block.split())
        if current and count + words > max_words:
            chunks.append("\n".join(current))
            # carry the tail of the previous chunk forward so no instruction loses its neighbours
            tail: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev.split()) > overlap_words:
                    break
                tail.insert(0, prev)
                carried += len(prev.split())
            current, count = tail, carried
        current.append(block)
        count += words
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_document(doc: Document, max_words: int = 170, overlap_words: int = 30) -> list[Chunk]:
    sections: list[tuple[str, list[str]]] = []
    heading, lines = "Overview", []
    for line in doc.body.splitlines():
        if line.startswith("## "):
            if any(s.strip() for s in lines):
                sections.append((heading, lines))
            heading, lines = line[3:].strip(), []
        elif not line.startswith("# "):
            lines.append(line)
    if any(s.strip() for s in lines):
        sections.append((heading, lines))

    chunks: list[Chunk] = []
    counter = 0
    for heading, lines in sections:
        blocks = [b.strip() for b in "\n".join(lines).split("\n") if b.strip()]
        kind = section_kind(heading)
        level_match = _LEVEL.search(heading)
        level = level_match.group(1).upper() if level_match else None
        # bullets in action sections stay whole; overlap only helps free prose
        overlap = 0 if kind != "context" else overlap_words
        for text in _pack(blocks, max_words, overlap):
            counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{counter}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=heading,
                    text=text,
                    kind=kind,
                    level=level,
                    tags=[str(t) for t in doc.metadata.get("hazard_tags", [])],
                    zone_tags=[str(t) for t in doc.metadata.get("zone_tags", [])],
                    metadata={k: doc.metadata.get(k, "") for k in ("source", "publisher", "jurisdiction", "doc_type", "reviewed")},
                )
            )
    return chunks


def corpus_fingerprint(documents: list[Document], *extra: Any) -> str:
    digest = hashlib.sha256()
    for doc in documents:
        digest.update(doc.doc_id.encode())
        digest.update(doc.body.encode())
        digest.update(repr(sorted(doc.metadata.items())).encode())
    digest.update(repr(extra).encode())
    return digest.hexdigest()[:16]
