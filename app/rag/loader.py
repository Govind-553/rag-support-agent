import os
import yaml
from pathlib import Path
from typing import List, Dict, Any

import datetime

class DocumentChunk:
    def __init__(self, doc_id: str, filename: str, heading: str, content: str, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.filename = filename
        self.heading = heading
        self.content = content
        self.metadata = self.clean_metadata(metadata)

    def clean_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively converts date/datetime objects to strings in metadata."""
        if not isinstance(meta, dict):
            return meta
        cleaned = {}
        for k, v in meta.items():
            if isinstance(v, (datetime.date, datetime.datetime)):
                cleaned[k] = v.isoformat()
            elif isinstance(v, dict):
                cleaned[k] = self.clean_metadata(v)
            elif isinstance(v, list):
                cleaned[k] = [
                    self.clean_metadata(x) if isinstance(x, dict) else (x.isoformat() if isinstance(x, (datetime.date, datetime.datetime)) else x)
                    for x in v
                ]
            else:
                cleaned[k] = v
        return cleaned

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "heading": self.heading,
            "content": self.content,
            "metadata": self.metadata
        }

def parse_markdown_file(file_path: Path) -> List[DocumentChunk]:
    """Parses a markdown file, extracts YAML front matter and chunks by headings."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    filename = file_path.name
    
    # 1. Parse YAML Front Matter
    metadata = {}
    content_start = 0
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except Exception as e:
                print(f"Error parsing front matter in {filename}: {e}")
            content_start = len(parts[0]) + 3 + len(parts[1]) + 3

    body = text[content_start:].strip()
    doc_id = metadata.get("document_id", filename)

    # 2. Chunk by headings
    chunks = []
    current_heading = "Introduction"
    current_lines = []

    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            # If we already have accumulated content, save it as a chunk
            current_content = "\n".join(current_lines).strip()
            if current_content:
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    filename=filename,
                    heading=current_heading,
                    content=current_content,
                    metadata=metadata
                ))
            # Set the new heading
            current_heading = stripped.lstrip("#").strip()
            current_lines = [line]  # Include the heading itself in the chunk content for semantic query matching
        else:
            current_lines.append(line)

    # Add the last chunk
    current_content = "\n".join(current_lines).strip()
    if current_content:
        chunks.append(DocumentChunk(
            doc_id=doc_id,
            filename=filename,
            heading=current_heading,
            content=current_content,
            metadata=metadata
        ))

    return chunks

def load_knowledge_base(kb_dir: Path) -> List[DocumentChunk]:
    """Loads and chunks all markdown files in the knowledge base directory."""
    all_chunks = []
    if not kb_dir.exists():
        return all_chunks

    for file in kb_dir.glob("*.md"):
        chunks = parse_markdown_file(file)
        all_chunks.extend(chunks)
        
    return all_chunks
