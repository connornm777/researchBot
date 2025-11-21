import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import numpy as np
from pdfminer.high_level import extract_text


class DataManager:
    """
    Central manager for:
      - metadata.json (per-PDF state, chunks, references)
      - embeddings.npy (vector store)
      - references.bib (BibTeX export)

    This is compatible with your existing data:
      - It reads the current metadata.json and embeddings.npy.
      - It uses existing 'chunks' entries and their 'embedding_index'.
      - It only changes embeddings when you call process_embeddings().

    Typical ingestion pipeline:

        dm = DataManager()
        dm.update_metadata()
        dm.process_pdfs()
        dm.chunk_text_files()
        dm.process_embeddings()
        dm.extract_references()
        dm.ensure_unique_citation_keys()
        dm.clean_metadata()
        dm.clean_metadata_references()
        dm.generate_references_bib()
    """

    # ------------------------------------------------------------------
    # init / paths / config
    # ------------------------------------------------------------------

    def __init__(self, data_server: Optional[str] = None) -> None:
        load_dotenv()

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=self.api_key)

        root = data_server or os.getenv("DATA_SERVER") or "."
        self.data_server = Path(root).expanduser()

        # core files
        self.metadata_file = self.data_server / "metadata.json"
        self.embedding_file = self.data_server / "embeddings.npy"
        self.reference_file = self.data_server / "references.bib"

        # directories
        self.pdf_files_directory = self.data_server / "pdfs"
        self.text_files_directory = self.data_server / "text"
        self.chunk_files_directory = self.data_server / "chunks"
        self.unscannable_pdfs_path = self.pdf_files_directory / "unscannable"

        for d in (
            self.pdf_files_directory,
            self.text_files_directory,
            self.chunk_files_directory,
            self.unscannable_pdfs_path,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # models / tokenizer
        self.tiktoken_model = os.getenv("TIKTOKEN_MODEL", "gpt-4o-mini")
        self.parsing_model = os.getenv("PARSING_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        self.encoder = tiktoken.encoding_for_model(self.tiktoken_model)

        # chunking config
        self.max_tokens_per_chunk = int(os.getenv("MAX_TOKENS_PER_CHUNK", "800"))
        self.overlap_tokens = int(os.getenv("OVERLAP_TOKENS", "80"))

        # batch size for embeddings (approx tokens_per_batch / tokens_per_chunk)
        batch_tokens = int(os.getenv("EMBEDDING_BATCH_TOKENS", "8000"))
        self.batch_size = max(1, batch_tokens // max(self.max_tokens_per_chunk, 1))

        # core state
        self.metadata: Dict[str, dict] = self.load_metadata()
        self.embeddings: np.ndarray = self.load_embeddings()

        # index: embedding_index -> (pdf_filename, chunk_idx)
        self.embedding_index: Dict[int, Tuple[str, int]] = {}
        self._build_embedding_index()

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------

    def _log(self, *args: Any) -> None:
        print("[DataManager]", *args)

    def token_count(self, text: str) -> int:
        return len(self.encoder.encode(text or ""))

    # ------------------------------------------------------------------
    # metadata / embeddings I/O
    # ------------------------------------------------------------------

    def load_metadata(self) -> Dict[str, dict]:
        if self.metadata_file.exists():
            try:
                with self.metadata_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                self._log("Warning: failed to load metadata.json:", e)
        return {}

    def save_metadata(self) -> None:
        try:
            with self.metadata_file.open("w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log("Error saving metadata.json:", e)

    def load_embeddings(self) -> np.ndarray:
        if self.embedding_file.exists():
            try:
                return np.load(self.embedding_file)
            except Exception as e:
                self._log("Warning: failed to load embeddings.npy:", e)
        return np.array([], dtype=np.float32)

    def save_embeddings(self) -> None:
        np.save(self.embedding_file, self.embeddings)

    # ------------------------------------------------------------------
    # embedding index
    # ------------------------------------------------------------------

    def _build_embedding_index(self) -> None:
        """
        Build lookup: embedding_index -> (pdf_filename, chunk_idx).

        Uses existing 'embedding_index' integers from metadata['chunks'].
        """
        self.embedding_index.clear()
        for pdf_filename, data in self.metadata.items():
            chunks = data.get("chunks") or []
            for chunk_idx, chunk_meta in enumerate(chunks):
                idx = chunk_meta.get("embedding_index")
                if isinstance(idx, int):
                    self.embedding_index[idx] = (pdf_filename, chunk_idx)

    # ------------------------------------------------------------------
    # metadata sync / cleaning
    # ------------------------------------------------------------------

    def update_metadata(self) -> None:
        """
        Discover new PDFs under pdfs/ and add skeleton entries to metadata.

        Does not remove existing entries.
        """
        pdf_files = [
            f for f in os.listdir(self.pdf_files_directory)
            if f.lower().endswith(".pdf")
        ]
        new_pdfs_found = False

        for pdf_filename in pdf_files:
            if pdf_filename not in self.metadata:
                self.metadata[pdf_filename] = {
                    "converted_to_text": False,
                    "text_filename": "",
                    "chunked": False,
                    "chunks": [],
                    "references": {},
                    "references_extracted": False,
                }
                self._log("Added new PDF to metadata:", pdf_filename)
                new_pdfs_found = True

        if new_pdfs_found:
            self.save_metadata()
            self._log("Metadata updated with new PDFs.")
        else:
            self._log("No new PDFs found.")

    def clean_metadata(self) -> None:
        """
        Remove legacy top-level fields now redundant with 'references'.
        """
        fields_to_remove = [
            "title",
            "authors",
            "affiliations",
            "embedded",
            "metadata_extracted",
        ]
        modified = False

        for pdf_filename, data in self.metadata.items():
            for field in fields_to_remove:
                if field in data:
                    del data[field]
                    modified = True
                    self._log(f"Removed '{field}' from metadata of {pdf_filename}")
            if "references" not in data:
                data["references"] = {}
                modified = True

        if modified:
            self.save_metadata()
            self._log("Metadata cleaned and saved.")
        else:
            self._log("Metadata already clean.")

    def clean_metadata_references(self) -> None:
        """
        Cheap deterministic cleanup of obvious junk from 'references'.
        """
        junk_markers = (
            "unknown",
            "not specified",
            "self-publishing",
            "draft last modified on",
        )
        changes_made = False

        for pdf_filename, data in self.metadata.items():
            refs = data.get("references") or {}
            if not isinstance(refs, dict):
                continue

            for key in list(refs.keys()):
                if key in ("type", "citation_key"):
                    continue
                value = refs.get(key)
                if not value:
                    continue
                s = str(value).lower()
                if any(m in s for m in junk_markers):
                    del refs[key]
                    changes_made = True
                    self._log(
                        f"Removed junk field '{key}' from references of {pdf_filename}"
                    )

        if changes_made:
            self.save_metadata()
            self._log("Cleaned metadata references and saved.")
        else:
            self._log("No junk fields found in references.")

    # ------------------------------------------------------------------
    # PDF -> text
    # ------------------------------------------------------------------

    def convert_pdf_to_text(self, pdf_filename: str) -> str:
        """
        Convert a PDF to plain text and return the text filename.
        """
        pdf_path = self.pdf_files_directory / pdf_filename
        text_filename = f"{Path(pdf_filename).stem}.txt"
        text_path = self.text_files_directory / text_filename

        self.text_files_directory.mkdir(parents=True, exist_ok=True)

        text = extract_text(str(pdf_path))
        with text_path.open("w", encoding="utf-8") as f:
            f.write(text)

        return text_filename

    def process_pdfs(self) -> None:
        """
        Convert any PDFs that haven't yet been converted to text.
        """
        changed = False
        for pdf_filename, data in self.metadata.items():
            if data.get("converted_to_text"):
                continue
            try:
                text_filename = self.convert_pdf_to_text(pdf_filename)
                data["converted_to_text"] = True
                data["text_filename"] = text_filename
                changed = True
                self._log(f"Converted {pdf_filename} -> {text_filename}")
            except Exception as e:
                self._log(f"Error converting {pdf_filename}:", e)

        if changed:
            self.save_metadata()

    # ------------------------------------------------------------------
    # text -> chunks
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks in token space.
        """
        tokens = self.encoder.encode(text or "")
        chunks: List[str] = []
        n = len(tokens)
        start = 0

        while start < n:
            end = min(n, start + max_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            if end == n:
                break
            start = max(0, end - overlap)

        return chunks

    def chunk_text_files(self) -> None:
        """
        Create chunk files for each converted text file that hasn't been chunked yet.
        """
        changed = False
        for pdf_filename, data in self.metadata.items():
            if not data.get("converted_to_text"):
                continue
            if data.get("chunked"):
                continue

            text_filename = data.get("text_filename")
            if not text_filename:
                continue

            text_path = self.text_files_directory / text_filename
            if not text_path.exists():
                self._log(f"Text file missing for {pdf_filename}: {text_path}")
                continue

            try:
                with text_path.open("r", encoding="utf-8") as f:
                    full_text = f.read()
            except Exception as e:
                self._log(f"Error reading {text_path}:", e)
                continue

            chunks = self.chunk_text(
                full_text,
                max_tokens=self.max_tokens_per_chunk,
                overlap=self.overlap_tokens,
            )

            base = Path(text_filename).stem
            self.chunk_files_directory.mkdir(parents=True, exist_ok=True)

            num_digits = len(str(max(len(chunks), 1)))
            chunk_metadata_list: List[dict] = []

            for idx, chunk in enumerate(chunks):
                chunk_filename = f"{base}_chunk{str(idx).zfill(num_digits)}.txt"
                chunk_path = self.chunk_files_directory / chunk_filename
                with chunk_path.open("w", encoding="utf-8") as f:
                    f.write(chunk)
                chunk_metadata_list.append(
                    {"filename": chunk_filename, "embedding_index": None}
                )

            data["chunked"] = True
            data["chunks"] = chunk_metadata_list
            changed = True
            self._log(f"Chunked {pdf_filename} into {len(chunks)} chunks.")

        if changed:
            self.save_metadata()

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Wrapper around OpenAI embeddings API.
        """
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [row.embedding for row in resp.data]

    def process_embeddings(self) -> None:
        """
        Generate embeddings for any chunks whose 'embedding_index' is None.

        Appends to existing embeddings.npy, preserving all prior vectors.
        """
        missing: List[Tuple[str, int, str]] = []
        for pdf_filename, data in self.metadata.items():
            if not data.get("chunked"):
                continue
            chunks = data.get("chunks") or []
            for chunk_idx, chunk_meta in enumerate(chunks):
                if isinstance(chunk_meta.get("embedding_index"), int):
                    continue
                chunk_filename = chunk_meta.get("filename")
                if not chunk_filename:
                    continue
                chunk_path = self.chunk_files_directory / chunk_filename
                try:
                    with chunk_path.open("r", encoding="utf-8") as f:
                        text = f.read()
                    missing.append((pdf_filename, chunk_idx, text))
                except Exception as e:
                    self._log(f"Error reading chunk {chunk_filename}:", e)

        if not missing:
            self._log("No new chunks to embed.")
            return

        total = len(missing)
        self._log(f"Embedding {total} chunks (batch_size={self.batch_size})...")

        for i in range(0, total, self.batch_size):
            batch = missing[i : i + self.batch_size]
            batch_texts = [x[2] for x in batch]

            try:
                batch_embeddings = np.array(
                    self.generate_embeddings(batch_texts),
                    dtype=np.float32,
                )
            except Exception as e:
                self._log(f"Error generating embeddings for batch {i}:", e)
                continue

            start_index = int(len(self.embeddings))
            if self.embeddings.size == 0:
                self.embeddings = batch_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, batch_embeddings])

            for j, (pdf_filename, chunk_idx, _text) in enumerate(batch):
                embedding_index = start_index + j
                chunk_meta = self.metadata[pdf_filename]["chunks"][chunk_idx]
                chunk_meta["embedding_index"] = embedding_index

        self.save_embeddings()
        self.save_metadata()
        self._build_embedding_index()

    # ------------------------------------------------------------------
    # reference extraction / BibTeX
    # ------------------------------------------------------------------

    def extract_references(self) -> None:
        """
        Use the first chunk of each paper to extract a reference entry and
        store it under metadata[pdf]['references'].
        """
        changed = False

        for pdf_filename, data in self.metadata.items():
            if not data.get("chunked"):
                continue
            if data.get("references_extracted"):
                continue

            chunks = data.get("chunks") or []
            if not chunks:
                continue

            first_chunk_filename = chunks[0].get("filename")
            if not first_chunk_filename:
                continue

            chunk_path = self.chunk_files_directory / first_chunk_filename
            try:
                with chunk_path.open("r", encoding="utf-8") as f:
                    chunk_text = f.read()
            except Exception as e:
                self._log(f"Error reading first chunk for {pdf_filename}:", e)
                continue

            system_msg = (
                "You are a bibliographic extraction assistant. "
                "Given the beginning of a scientific paper, you must output a single "
                "JSON object describing one BibTeX-like entry for that paper. "
                "Use fields: type, citation_key, title, author, year, journal, "
                "booktitle, publisher, volume, number, pages, doi, url. "
                "If something is unknown, use an empty string. "
                "The 'author' field should follow BibTeX style "
                "'Last, First and SecondLast, SecondFirst ...'. "
                "The citation_key should be a short identifier like "
                "lastnameYearShortTitle (no spaces). "
                "Respond with JSON only."
            )

            user_msg = (
                f"PDF filename: {pdf_filename}\n\n"
                f"First chunk of text:\n\n{chunk_text[:6000]}"
            )

            try:
                resp = self.client.chat.completions.create(
                    model=self.parsing_model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                content = resp.choices[0].message.content or "{}"
                ref_data = json.loads(content)

                if not isinstance(ref_data, dict):
                    self._log(f"Non-dict references for {pdf_filename}, skipping.")
                    continue

                ref_data.setdefault("type", "misc")
                if not ref_data.get("citation_key"):
                    stem = Path(pdf_filename).stem.replace(" ", "_")
                    ref_data["citation_key"] = stem

                data["references"] = ref_data
                data["references_extracted"] = True
                changed = True
                self._log(f"Extracted references for {pdf_filename}.")

            except Exception as e:
                self._log(f"Error extracting references for {pdf_filename}:", e)

        if changed:
            self.save_metadata()

    def ensure_unique_citation_keys(self) -> None:
        """
        Ensure each reference entry has a unique citation_key. If collisions
        occur, append -2, -3, ... to later ones.
        """
        key_map: Dict[str, List[str]] = {}
        for pdf_filename, data in self.metadata.items():
            refs = data.get("references") or {}
            key = refs.get("citation_key")
            if not key:
                continue
            key_map.setdefault(key, []).append(pdf_filename)

        changed = False
        for key, pdfs in key_map.items():
            if len(pdfs) <= 1:
                continue
            for i, pdf_filename in enumerate(pdfs[1:], start=2):
                refs = self.metadata[pdf_filename].get("references") or {}
                new_key = f"{key}-{i}"
                refs["citation_key"] = new_key
                changed = True
                self._log(
                    f"Renamed citation_key for {pdf_filename}: {key} -> {new_key}"
                )

        if changed:
            self.save_metadata()
            self._log("Ensured uniqueness of citation keys.")
        else:
            self._log("No duplicate citation keys found.")

    @staticmethod
    def _escape_bibtex_value(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("{", "\\{")
            .replace("}", "\\}")
        )

    def generate_references_bib(self) -> None:
        """
        Write references.bib based on metadata['references'].
        """
        try:
            with self.reference_file.open("w", encoding="utf-8") as f:
                for pdf_filename, data in self.metadata.items():
                    refs = data.get("references") or {}
                    if not refs:
                        continue

                    entry_type = refs.get("type", "misc")
                    citation_key = refs.get("citation_key")
                    if not citation_key:
                        stem = Path(pdf_filename).stem.replace(" ", "_")
                        citation_key = stem
                        refs["citation_key"] = citation_key

                    f.write(f"@{entry_type}{{{citation_key},\n")
                    for key, raw_value in refs.items():
                        if key in ("type", "citation_key"):
                            continue
                        if raw_value is None or raw_value == "":
                            continue
                        if isinstance(raw_value, list):
                            value = " and ".join(map(str, raw_value))
                        else:
                            value = str(raw_value)
                        value = self._escape_bibtex_value(value)
                        f.write(f"  {key} = {{{value}}},\n")
                    f.write("}\n\n")

            self._log(f"Generated {self.reference_file}.")
        except Exception as e:
            self._log("Error writing references.bib:", e)

    # ------------------------------------------------------------------
    # searching
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Vector search over chunk embeddings.

        Returns a list of dicts:
            {
                "pdf_filename": str,
                "chunk_filename": str,
                "similarity_score": float,
                "references": dict,
            }
        """
        top_k = max(1, int(top_k))

        if self.embeddings.size == 0:
            self._log("No embeddings available for search.")
            return []

        query_embedding = np.array(
            self.generate_embeddings([query])[0],
            dtype=np.float32,
        )
        scores = self.embeddings @ query_embedding

        k = min(top_k, scores.shape[0])
        top_indices = np.argsort(scores)[-k:][::-1]

        results: List[dict] = []
        for emb_idx in top_indices:
            emb_idx_int = int(emb_idx)
            mapping = self.embedding_index.get(emb_idx_int)
            if mapping is None:
                continue
            pdf_filename, chunk_idx = mapping
            data = self.metadata.get(pdf_filename) or {}
            chunks = data.get("chunks") or []
            if not (0 <= chunk_idx < len(chunks)):
                continue
            chunk_meta = chunks[chunk_idx]
            chunk_filename = chunk_meta.get("filename")
            if not chunk_filename:
                continue

            results.append(
                {
                    "pdf_filename": pdf_filename,
                    "chunk_filename": chunk_filename,
                    "similarity_score": float(scores[emb_idx]),
                    "references": data.get("references", {}),
                }
            )

        return results

    # ------------------------------------------------------------------
    # destructive operations
    # ------------------------------------------------------------------

    def remove_pdf_entry(self, pdf_filename: str) -> None:
        """
        Remove a PDF from metadata, move the PDF to pdfs/unscannable/,
        delete associated text/chunk files, remove its embeddings, and
        reindex the remaining embeddings and metadata.
        """
        if pdf_filename not in self.metadata:
            self._log(f"No metadata found for {pdf_filename}. Nothing to remove.")
            return

        data = self.metadata[pdf_filename]
        chunks = data.get("chunks") or []

        # collect embedding indices
        to_remove_indices = sorted(
            idx
            for idx in (
                ch.get("embedding_index") for ch in chunks
            )
            if isinstance(idx, int)
        )

        # move PDF
        pdf_path = self.pdf_files_directory / pdf_filename
        self.unscannable_pdfs_path.mkdir(parents=True, exist_ok=True)
        if pdf_path.exists():
            dest = self.unscannable_pdfs_path / pdf_filename
            pdf_path.replace(dest)
            self._log(f"Moved {pdf_filename} to {dest}")
        else:
            self._log(f"PDF file not found on disk: {pdf_path}")

        # remove text file
        text_filename = data.get("text_filename")
        if text_filename:
            text_path = self.text_files_directory / text_filename
            if text_path.exists():
                text_path.unlink()
                self._log(f"Deleted text file {text_path}")

        # remove chunk files
        for ch in chunks:
            chunk_filename = ch.get("filename")
            if not chunk_filename:
                continue
            chunk_path = self.chunk_files_directory / chunk_filename
            if chunk_path.exists():
                chunk_path.unlink()
                self._log(f"Deleted chunk file {chunk_path}")

        # remove metadata entry
        del self.metadata[pdf_filename]

        # remove embeddings and reindex
        if self.embeddings.size > 0 and to_remove_indices:
            old_count = len(self.embeddings)
            keep_mask = np.ones(old_count, dtype=bool)
            for idx in to_remove_indices:
                if 0 <= idx < old_count:
                    keep_mask[idx] = False

            new_embeddings = self.embeddings[keep_mask]

            # map old index -> new index (or None if dropped)
            old_to_new: Dict[int, Optional[int]] = {}
            new_idx = 0
            for i in range(old_count):
                if keep_mask[i]:
                    old_to_new[i] = new_idx
                    new_idx += 1
                else:
                    old_to_new[i] = None

            # update embedding_index stored in chunks
            for pdf, mdata in self.metadata.items():
                for ch in mdata.get("chunks") or []:
                    old_idx = ch.get("embedding_index")
                    if not isinstance(old_idx, int):
                        continue
                    new_val = old_to_new.get(old_idx)
                    ch["embedding_index"] = new_val

            self.embeddings = new_embeddings
            self.save_embeddings()

        self.save_metadata()
        self._build_embedding_index()
        self._log(
            f"Removed {pdf_filename}, reindexed embeddings, and saved metadata."
        )


def run_pipeline(dm: DataManager) -> None:
    """
    Convenience pipeline: new PDFs -> text -> chunks -> embeddings.
    """
    dm.update_metadata()
    dm.process_pdfs()
    dm.chunk_text_files()
    dm.process_embeddings()
    dm._log("Pipeline completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataManager pipeline helper")
    parser.add_argument(
        "command",
        nargs="?",
        default="pipeline",
        choices=["pipeline", "update", "embed", "bib", "clean"],
        help="pipeline|update|embed|bib|clean",
    )
    args = parser.parse_args()

    dm = DataManager()

    if args.command == "pipeline":
        run_pipeline(dm)
        dm.extract_references()
        dm.ensure_unique_citation_keys()
        dm.clean_metadata()
        dm.clean_metadata_references()
        dm.generate_references_bib()
    elif args.command == "update":
        dm.update_metadata()
        dm.process_pdfs()
        dm.chunk_text_files()
    elif args.command == "embed":
        dm.process_embeddings()
    elif args.command == "bib":
        dm.extract_references()
        dm.ensure_unique_citation_keys()
        dm.clean_metadata()
        dm.clean_metadata_references()
        dm.generate_references_bib()
    elif args.command == "clean":
        dm.clean_metadata()
        dm.clean_metadata_references()
