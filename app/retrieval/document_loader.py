"""Document loader module for NLP educational dataset collections.

Loads, parses, and validates the 6 educational document collections:
1. Text Preprocessing and Tokenization
2. Sentiment Analysis and Text Classification
3. Word Embeddings and Language Models
4. Named Entity Recognition and Information Extraction
5. RAG, Semantic Search and Question Answering
6. Machine Translation and Text Summarization
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default path to collections directory relative to project root
DEFAULT_COLLECTIONS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "nlp_collections"


@dataclass
class PassageChunk:
    """Represents an individual indexed passage / page of an educational document."""
    document_id: str
    chunk_id: str
    page_number: int
    title: str
    category: str
    source: str
    section_title: str
    text: str

    def to_retrieval_dict(self) -> Dict[str, Any]:
        """Convert to the standard chunk dictionary expected by Person 3's generation layer."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "text": self.text,
            "title": self.title,
            "category": self.category,
            "source": self.source,
            "section_title": self.section_title,
        }


@dataclass
class Document:
    """Represents a full educational document containing multiple pages / chunks."""
    id: str
    title: str
    category: str
    source: str
    chunks: List[PassageChunk] = field(default_factory=list)


@dataclass
class NLPCollection:
    """Represents a collection of educational documents for a specific NLP domain."""
    collection_id: str
    collection_name: str
    description: str
    documents: List[Document] = field(default_factory=list)


class DocumentLoader:
    """Loads and manages all 6 NLP educational dataset collections from disk."""

    CANONICAL_COLLECTIONS = [
        "Text Preprocessing and Tokenization",
        "Sentiment Analysis and Text Classification",
        "Word Embeddings and Language Models",
        "Named Entity Recognition and Information Extraction",
        "RAG, Semantic Search and Question Answering",
        "Machine Translation and Text Summarization",
    ]

    def __init__(self, collections_dir: Optional[Path] = None):
        self.collections_dir = Path(collections_dir or DEFAULT_COLLECTIONS_DIR)
        self.collections: Dict[str, NLPCollection] = {}
        self.documents: Dict[str, Document] = {}
        self.chunks: List[PassageChunk] = []
        self._load_all()

    def _load_all(self) -> None:
        """Scan directory and load all JSON collection files."""
        if not self.collections_dir.exists():
            logger.warning(f"Collections directory not found at {self.collections_dir}")
            return

        json_files = sorted(self.collections_dir.glob("*.json"))
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                col_id = data.get("collection_id", file_path.stem)
                col_name = data.get("collection_name", col_id)
                col_desc = data.get("description", "")

                collection = NLPCollection(
                    collection_id=col_id,
                    collection_name=col_name,
                    description=col_desc,
                )

                for doc_data in data.get("documents", []):
                    doc_id = doc_data["id"]
                    doc_title = doc_data["title"]
                    doc_category = doc_data.get("category", col_name)
                    doc_source = doc_data.get("source", "Unknown Source")

                    doc = Document(
                        id=doc_id,
                        title=doc_title,
                        category=doc_category,
                        source=doc_source,
                    )

                    for page_data in doc_data.get("pages", []):
                        chunk = PassageChunk(
                            document_id=doc_id,
                            chunk_id=page_data.get("chunk_id", f"{doc_id}_p{page_data.get('page_number', 1)}"),
                            page_number=int(page_data.get("page_number", 1)),
                            title=doc_title,
                            category=doc_category,
                            source=doc_source,
                            section_title=page_data.get("section_title", ""),
                            text=page_data.get("text", "").strip(),
                        )
                        doc.chunks.append(chunk)
                        self.chunks.append(chunk)

                    collection.documents.append(doc)
                    self.documents[doc_id] = doc

                self.collections[col_name] = collection
            except Exception as e:
                logger.error(f"Error loading collection file {file_path}: {e}")

    def get_collection_names(self) -> List[str]:
        """Return list of all available collection names."""
        if self.collections:
            return list(self.collections.keys())
        return self.CANONICAL_COLLECTIONS

    def get_collection(self, name: str) -> Optional[NLPCollection]:
        """Get collection by exact or case-insensitive name."""
        if name in self.collections:
            return self.collections[name]
        lowered = name.strip().lower()
        for k, v in self.collections.items():
            if k.lower() == lowered or v.collection_id.lower() == lowered:
                return v
        return None

    def get_chunks(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all passage chunks, optionally restricted to a specific collection.

        Parameters:
            collection_name: If specified (and not 'All Collections'), restricts
                             chunks strictly to that collection category.
        """
        if not collection_name or collection_name.strip() in ("", "All Collections", "All Datasets"):
            return [c.to_retrieval_dict() for c in self.chunks]

        col = self.get_collection(collection_name)
        if col:
            res = []
            for doc in col.documents:
                for chunk in doc.chunks:
                    res.append(chunk.to_retrieval_dict())
            return res

        # Fallback category match
        lowered = collection_name.strip().lower()
        return [
            c.to_retrieval_dict()
            for c in self.chunks
            if c.category.lower() == lowered
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return statistical summary of loaded collections."""
        return {
            "total_collections": len(self.collections),
            "total_documents": len(self.documents),
            "total_passages": len(self.chunks),
            "collection_breakdown": {
                name: {
                    "document_count": len(col.documents),
                    "passage_count": sum(len(d.chunks) for d in col.documents),
                }
                for name, col in self.collections.items()
            },
        }


# Global singleton instance for easy access
_global_loader: Optional[DocumentLoader] = None


def get_document_loader() -> DocumentLoader:
    """Get or initialize the singleton DocumentLoader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = DocumentLoader()
    return _global_loader
