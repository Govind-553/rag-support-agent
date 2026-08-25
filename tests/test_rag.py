import pytest
from pathlib import Path
import tempfile
import shutil
from app.rag.loader import parse_markdown_file, load_knowledge_base
from app.rag.index import FAISSIndex
from app.config import KNOWLEDGE_BASE_DIR

def test_parse_markdown_file():
    # Create a temporary markdown file to test parsing
    content = """---
document_id: TEST-123
title: Test Document
status: active
effective_date: 2026-01-01
---
# Main Header

Some intro text here.

## Subsection A

Details about Subsection A.

## Subsection B

Details about Subsection B.
"""
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        chunks = parse_markdown_file(temp_path)
        assert len(chunks) == 3
        
        # In the parsed chunks:
        # First chunk
        assert chunks[0].doc_id == "TEST-123"
        assert chunks[0].heading == "Main Header"
        assert "Some intro text here." in chunks[0].content
        assert chunks[0].metadata["title"] == "Test Document"
        assert chunks[0].metadata["status"] == "active"
        
        # Second chunk
        assert chunks[1].heading == "Subsection A"
        assert "Details about Subsection A." in chunks[1].content
        
        # Third chunk
        assert chunks[2].heading == "Subsection B"
        assert "Details about Subsection B." in chunks[2].content
    finally:
        temp_path.unlink()

def test_load_knowledge_base():
    # Test loading actual knowledge base
    chunks = load_knowledge_base(KNOWLEDGE_BASE_DIR)
    assert len(chunks) > 0
    # Verify metadata is preserved
    for chunk in chunks:
        assert chunk.filename.endswith(".md")
        assert "status" in chunk.metadata
        assert "document_id" in chunk.metadata

def test_faiss_index_build_and_search():
    # Build FAISS index from actual knowledge base
    chunks = load_knowledge_base(KNOWLEDGE_BASE_DIR)
    assert len(chunks) > 0

    with tempfile.TemporaryDirectory() as temp_dir:
        index_dir = Path(temp_dir) / "faiss_index"
        faiss_index = FAISSIndex(index_dir=index_dir)
        
        # Build index
        faiss_index.build(chunks)
        
        # Verify index files were written
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "chunks.json").exists()
        
        # Load and search
        new_index = FAISSIndex(index_dir=index_dir)
        loaded = new_index.load()
        assert loaded is True
        
        # Search query
        results = new_index.search("How long to return a bag?", k=2)
        assert len(results) > 0
        
        # Verify results structure
        chunk, score = results[0]
        assert chunk.filename.endswith(".md")
        assert isinstance(score, float)
        # Cosine similarity of normalized vectors should be in [-1, 1]
        assert -1.0 <= score <= 1.0
