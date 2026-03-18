import os
import io
import csv
import json
import threading
from typing import List, Dict, Any, Optional
from backend.config import settings
from backend.utils import logger

# In-memory job tracker for background ingestion tasks
_jobs: Dict[str, Dict[str, Any]] = {}

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    return _jobs.get(job_id)

def set_job_status(job_id: str, status: str, message: str = "", chunks: int = 0):
    _jobs[job_id] = {"status": status, "message": message, "chunks": chunks}


# ── parsers ──────────────────────────────────────────────────────────────────

def extract_text_from_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


def extract_text_from_pdf(content: bytes) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        raise ValueError("PyMuPDF not installed. Run: pip install pymupdf")
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")


def extract_text_from_docx(content: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)
    except ImportError:
        raise ValueError("python-docx not installed. Run: pip install python-docx")
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX: {e}")


def extract_text_from_md(content: bytes) -> str:
    """Markdown files are already plain text; strip basic markers."""
    text = content.decode("utf-8", errors="ignore")
    return text


def extract_text_from_csv(content: bytes) -> str:
    """Convert each CSV row into a human-readable sentence."""
    lines = []
    reader = csv.DictReader(io.StringIO(content.decode("utf-8", errors="ignore")))
    for row in reader:
        lines.append(", ".join(f"{k}: {v}" for k, v in row.items() if v))
    return "\n".join(lines)


def extract_text_from_json(content: bytes) -> str:
    """Recursively flatten a JSON document into readable key-value lines."""
    text = content.decode("utf-8", errors="ignore")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text  # return raw if not valid JSON

    lines: List[str] = []

    def _flatten(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _flatten(v, f"{prefix}[{i}]")
        else:
            lines.append(f"{prefix}: {obj}")

    _flatten(data)
    return "\n".join(lines)


# ── supported extensions ──────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".csv", ".json"}

PARSERS = {
    ".txt":  extract_text_from_txt,
    ".pdf":  extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".md":   extract_text_from_md,
    ".csv":  extract_text_from_csv,
    ".json": extract_text_from_json,
}


# ── DocumentLoader ───────────────────────────────────────────────────────────

class DocumentLoader:
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    # --- public interface ---

    def parse_file(self, filename: str, content: bytes) -> str:
        """
        Auto-detect file type from extension and return extracted plain text.
        Raises ValueError for unsupported types.
        """
        ext = os.path.splitext(filename.lower())[1]
        if ext not in PARSERS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        logger.info(f"Parsing '{filename}' as {ext}")
        return PARSERS[ext](content)

    def process_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Split a document's text into chunks.
        Returns a list of dicts: {text, metadata}.
        """
        if not text:
            return []

        chunks = self._recursive_split(text)

        processed: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            clean = chunk.strip().replace("\n", " ")
            if not clean:
                continue
            processed.append({
                "text": clean,
                "metadata": {"source": source, "chunk_id": i}
            })

        logger.info(f"'{source}' → {len(processed)} chunks")
        return processed

    def ingest_file(self, filename: str, content: bytes, source: str) -> List[Dict[str, Any]]:
        """
        Convenience: parse a file and return chunks ready for embedding.
        """
        text = self.parse_file(filename, content)
        return self.process_text(text, source)

    # --- chunking machinery ---

    def _recursive_split(self, text: str) -> List[str]:
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._split_text_with_separators(text, separators)

    def _split_text_with_separators(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        for s in separators[:-1]:
            if s in text:
                separator = s
                break

        splits = text.split(separator) if separator else list(text)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split) + (len(separator) if separator else 0)

            if current_length + split_len > self.chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                # overlap
                overlap, olen = [], 0
                for item in reversed(current_chunk):
                    ilen = len(item) + (len(separator) if separator else 0)
                    if olen + ilen > self.chunk_overlap:
                        break
                    overlap.insert(0, item)
                    olen += ilen
                current_chunk = list(overlap)
                current_length = olen

            if split_len > self.chunk_size:
                if len(separators) > 1:
                    sub = self._split_text_with_separators(split, separators[1:])
                    for s in sub[:-1]:
                        chunks.append(s)
                    current_chunk = [sub[-1]]
                    current_length = len(sub[-1])
                else:
                    for i in range(0, len(split), self.chunk_size):
                        chunks.append(split[i:i + self.chunk_size])
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(split)
                current_length += split_len

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks


document_loader = DocumentLoader()
