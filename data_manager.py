import os, pdb
import json
import openai
import tiktoken
from dotenv import dotenv_values
import numpy as np
from pdfminer.high_level import extract_text
from PyPDF2 import PdfReader

class DataManager:

    def __init__(self):
        # Load environment variables from .env file
        for key, value in dotenv_values().items():
            setattr(self, key, value)

        # Set OpenAI API key
        openai.api_key = getattr(self, 'OPENAI_API_KEY', None)

        # Set models and encoder
        self.TIKTOKEN_MODEL = getattr(self, 'TIKTOKEN_MODEL', 'gpt-3.5-turbo')
        self.PARSE_MODEL = getattr(self, 'PARSE_MODEL', self.TIKTOKEN_MODEL)
        self.EMB_MODEL = getattr(self, 'EMB_MODEL', 'text-embedding-ada-002')
        self.encoder = tiktoken.encoding_for_model(self.TIKTOKEN_MODEL)

        # Convert parameters to appropriate types
        self.MAX_TOKENS_PER_CHUNK = int(self.MAX_TOKENS_PER_CHUNK)
        self.OVERLAP_TOKENS = int(self.OVERLAP_TOKENS)
        # Batch size (adjust as appropriate)
        self.batch_size = int(8000/self.MAX_TOKENS_PER_CHUNK)
        # Initialize metadata and embeddings
        self.metadata = self.load_metadata()
        self.embeddings = self.load_embeddings()

    def token_count(self, text):
        return len(self.encoder.encode(text))

    def load_metadata(self):
        if os.path.exists(self.META_PATH):
            with open(self.META_PATH, 'r') as f:
                return json.load(f)
        else:
            return {}

    def save_metadata(self):
        with open(self.META_PATH, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def load_embeddings(self):
        if os.path.exists(self.EMB_PATH):
            embeddings = np.load(self.EMB_PATH)
            return embeddings
        else:
            return np.array([])

    def save_embeddings(self):
        np.save(self.EMB_PATH, self.embeddings)

    def update_metadata(self):
        """
        Updates the metadata by adding any new PDFs found in the PDF_PATH directory.
        Only PDFs listed in the metadata will be processed by other functions.
        """
        pdf_files = [f for f in os.listdir(self.PDF_PATH) if f.endswith('.pdf')]
        new_pdfs_found = False

        for pdf_filename in pdf_files:
            if pdf_filename not in self.metadata:
                # Initialize metadata for new PDF
                self.metadata[pdf_filename] = {
                    'converted_to_text': False,
                    'text_filename': '',
                    'chunked': False,
                    'chunks': [],
                    'references': {},  # Open-ended sub-dictionary for reference info
                    'references_extracted': False,
                    # Other processing flags can be added as needed
                }
                print(f"Added new PDF to metadata: {pdf_filename}")
                new_pdfs_found = True

        if new_pdfs_found:
            self.save_metadata()
            print("Metadata updated with new PDFs.")
        else:
            print("No new PDFs found.")

    def clean_metadata(self):
        """
        Removes redundant author, title, and affiliations data not under the 'references'
        sub-dictionary in the metadata.
        """
        fields_to_remove = ['title', 'authors', 'affiliations', 'embedded', 'metadata_extracted']
        modified = False

        for pdf_filename, data in self.metadata.items():
            for field in fields_to_remove:
                if field in data:
                    del data[field]
                    modified = True
                    print(f"Removed '{field}' from metadata of {pdf_filename}")
            # Ensure 'references' sub-dictionary exists
            if 'references' not in data:
                data['references'] = {}
                modified = True

        if modified:
            self.save_metadata()
            print("Metadata cleaned and saved.")
        else:
            print("No redundant fields found. Metadata is clean.")

    def process_pdfs(self):
        """
        Processes PDFs listed in the metadata.
        Only PDFs that are in the metadata will be processed.
        """
        for pdf_filename in self.metadata:
            if self.metadata[pdf_filename]['converted_to_text']:
                print(f"Text for {pdf_filename} already exists. Skipping conversion.")
                continue

            # Convert PDF to text
            try:
                text_filename = self.convert_pdf_to_text(pdf_filename)
                self.metadata[pdf_filename]['converted_to_text'] = True
                self.metadata[pdf_filename]['text_filename'] = text_filename
                self.save_metadata()
                print(f"Converted {pdf_filename} to text.")
            except Exception as e:
                print(f"Error converting {pdf_filename} to text: {e}")

    def convert_pdf_to_text(self, pdf_filename):
        pdf_path = os.path.join(self.PDF_PATH, pdf_filename)
        text_filename = os.path.splitext(pdf_filename)[0] + '.txt'
        text_path = os.path.join(self.TXT_PATH, text_filename)

        os.makedirs(self.TXT_PATH, exist_ok=True)

        text = extract_text(pdf_path)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return text_filename  # Return the name of the text file created

    def chunk_text(self, text, max_tokens=2000, overlap=50):
        """
        Splits a string into chunks with overlap, each containing up to max_tokens tokens.

        Args:
            text (str): The input text string to be chunked.
            max_tokens (int): The maximum number of tokens per chunk.
            overlap (int): The number of tokens to overlap between chunks.

        Returns:
            List[str]: A list of text chunks.
        """
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += max_tokens - overlap  # Move the start position forward with overlap
        return chunks

    def chunk_text_files(self):
        """
        Processes text files listed in the metadata, creates chunk files with overlap,
        and updates the metadata accordingly.

        Only processes text files that have not been chunked yet.
        """
        for pdf_filename, data in self.metadata.items():
            if not data.get('converted_to_text'):
                print(f"PDF {pdf_filename} has not been converted to text yet. Skipping.")
                continue

            if data.get('chunked'):
                print(f"Text for {pdf_filename} already chunked. Skipping.")
                continue

            text_filename = data['text_filename']
            text_path = os.path.join(self.TXT_PATH, text_filename)
            base_filename = os.path.splitext(text_filename)[0]
            os.makedirs(self.CHUNK_PATH, exist_ok=True)

            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Use chunk_text to get the chunks
                chunks = self.chunk_text(
                    text,
                    max_tokens=self.MAX_TOKENS_PER_CHUNK,
                    overlap=self.OVERLAP_TOKENS
                )

                # Write chunks to disk with appropriate padding
                num_digits = len(str(len(chunks)))
                chunk_metadata_list = []
                for idx, chunk in enumerate(chunks):
                    chunk_filename = f"{base_filename}_chunk{str(idx).zfill(num_digits)}.txt"
                    chunk_file_path = os.path.join(self.CHUNK_PATH, chunk_filename)
                    with open(chunk_file_path, 'w', encoding='utf-8') as f:
                        f.write(chunk)
                    chunk_metadata = {
                        'filename': chunk_filename,
                        'embedding_index': None
                    }
                    chunk_metadata_list.append(chunk_metadata)

                # Update metadata
                data['chunked'] = True
                data['chunks'] = chunk_metadata_list
                self.save_metadata()

                print(f"Chunked text for {pdf_filename} into {len(chunks)} chunks.")
            except Exception as e:
                print(f"Error chunking text for {pdf_filename}: {e}")

    def remove_problematic_entries(self):
        """
        Identify and remove problematic entries from metadata and embeddings.
        For example, entries with empty titles.
        This function:
        - Finds entries with empty (or invalid) titles.
        - Asks the user to confirm deletion.
        - Removes those entries and their associated embeddings from self.embeddings.
        - Reindexes embedding indexes for all remaining entries.
        """

        # Identify problematic PDFs (for example, those with empty titles)
        # You can adjust this condition as needed
        problematic_pdfs = []
        for pdf_filename, data in self.metadata.items():
            references = data.get('references', {})
            title = references.get('title', '').strip()
            # Condition: empty title or no title field
            if title == '':
                problematic_pdfs.append(pdf_filename)

        if not problematic_pdfs:
            print("No problematic entries found (e.g., no empty titles).")
            return

        # Ask for confirmation to remove each problematic PDF
        to_remove = []
        for pdf_filename in problematic_pdfs:
            answer = input(
                f"Found problematic entry: {pdf_filename} has empty title. Remove it? (y/n): ").strip().lower()
            if answer == 'y':
                to_remove.append(pdf_filename)
            else:
                print(f"Skipping removal of {pdf_filename}.")

        if not to_remove:
            print("No entries selected for removal.")
            return

        # Gather all embeddings to remove
        # We'll collect all embedding indexes from each PDF we're removing
        embedding_indexes_to_remove = []
        for pdf_filename in to_remove:
            if self.metadata[pdf_filename]['converted_to_text']:
                os.remove(os.path.join(self.TXT_PATH, self.metadata[pdf_filename]['text_filename']))
            os.replace(os.path.join(self.PDF_PATH, pdf_filename), os.path.join(self.MAN_PATH, pdf_filename))
            data = self.metadata[pdf_filename]
            # Collect all embedding indexes from the chunks
            chunks = data.get('chunks', [])
            for chunk_meta in chunks:
                idx = chunk_meta.get('embedding_index')
                if idx is not None:
                    embedding_indexes_to_remove.append(idx)

        if not embedding_indexes_to_remove and not to_remove:
            # Nothing to remove
            print("No embeddings or entries to remove.")
            return

        # Sort embedding indexes to remove
        embedding_indexes_to_remove.sort()

        # Create a mask for embeddings we want to keep
        old_count = len(self.embeddings)
        keep_mask = np.ones(old_count, dtype=bool)
        for idx in embedding_indexes_to_remove:
            if 0 <= idx < old_count:
                keep_mask[idx] = False

        # Filter embeddings
        new_embeddings = self.embeddings[keep_mask]

        # Now we need to update embedding_index references in metadata
        # Create a mapping from old indexes to new indexes
        old_to_new = [None] * old_count
        new_index = 0
        for i in range(old_count):
            if keep_mask[i]:
                old_to_new[i] = new_index
                new_index += 1

        # Remove the PDFs from metadata and fix embedding indexes in remaining entries
        for pdf_filename in to_remove:
            del self.metadata[pdf_filename]

        # Update all embedding indexes in remaining metadata
        for pdf_filename, data in self.metadata.items():
            chunks = data.get('chunks', [])
            for chunk_meta in chunks:
                idx = chunk_meta.get('embedding_index', None)
                if idx is not None and idx < old_count:
                    new_idx = old_to_new[idx]
                    # If this embedding was removed, new_idx would be None, but that
                    # should not happen for chunks we keep. Just in case:
                    if new_idx is None:
                        # This chunk lost its embedding, handle as needed or set to None
                        chunk_meta['embedding_index'] = None
                    else:
                        chunk_meta['embedding_index'] = new_idx

        # Assign updated embeddings back to self.embeddings
        self.embeddings = new_embeddings
        self.save_embeddings()
        self.save_metadata()
        print("Removed selected entries and reorganized embeddings and metadata indexes.")

    def clear_references(self):
        """
        Clears old references from the metadata, resets the 'references_extracted' flag,
        and optionally removes the existing references.bib file.
        """
        # Iterate over all PDFs in the metadata
        for pdf_filename, data in self.metadata.items():
            # Clear references
            data['references'] = {}
            # Reset the flag so that extract_references can run again
            data['references_extracted'] = False

        # Save updated metadata
        self.save_metadata()
        print("Cleared old references in metadata and reset 'references_extracted' flags.")

        # Optionally remove the existing references.bib file if you want a fresh start
        # Comment this out if you prefer to keep the old file
        if os.path.exists(self.REF_PATH):
            os.remove(self.REF_PATH)
            print(f"Removed old {self.REF_PATH} file.")

    def extract_references(self):
        """
        Uses GPT to extract reference information from the first chunk
        of each text file, and updates the metadata under the 'references' field.
        Also requests the model to provide a suitable citation key and item type.
        """
        for pdf_filename, data in self.metadata.items():
            if not data.get('chunked'):
                print(f"Text for {pdf_filename} has not been chunked yet. Skipping metadata extraction.")
                continue

            if data.get('references_extracted'):
                print(f"References for {pdf_filename} already extracted. Skipping.")
                continue

            try:
                # Get the first chunk metadata
                first_chunk_meta = data['chunks'][0]
                chunk_filename = first_chunk_meta['filename']
                chunk_file_path = os.path.join(self.CHUNK_PATH, chunk_filename)
                with open(chunk_file_path, 'r', encoding='utf-8') as f:
                    first_chunk_text = f.read()

                # Prepare the prompt for GPT
                # We now ask for 'citation_key' and 'type' fields as well.
                prompt = f"""
    Extract bibliographic information from the text below and return it as a JSON object that can be directly converted into a valid BibTeX entry.

    Fields to use if present:
    - title
    - author (as a list of strings, each "FirstName LastName")
    - journal
    - year
    - volume
    - number
    - pages
    - publisher
    - address
    - note
    - doi
    - url

    If an author uses an equation in the title, make sure you use $ to enclose it so it compiles in latex.

    All values must be strings. If a field is unknown, omit it. Don't write 'none' or 'not specified'. Do not include any fields not listed above.

    Additionally, determine the appropriate BibTeX item type:
    - If 'journal' is present, use "article".
    - If 'publisher' and 'address' are present and no 'journal', use "book".
    - Otherwise, use "misc".

    Return a 'type' field indicating the chosen item type.

    Also, generate a 'citation_key' field that is:
    - all lowercase
    - no special characters or spaces
    - as short as possible but still unique
    - derive it from the first author's last name (if available), the year (if available), and a distinctive short portion of the title (if available)
    - If any of these are missing, just do your best to create a short stable key.

    Return only these fields and do not include commentary or additional text outside the JSON.

    Example JSON:
    {{
      "type": "article",
      "citation_key": "hallwightman1957",
      "title": "A Theorem on Invariant Analytic Functions with Applications to Relativistic Quantum Field Theory",
      "author": ["D. Hall", "A. S. Wightman"],
      "journal": "Matematisk-fysiske Meddelelser",
      "volume": "31",
      "number": "5",
      "year": "1957",
      "publisher": "Det Kongelige Danske Videnskabernes Selskab",
      "address": "Copenhagen",
      "note": "In commission at Ejnar Munksgaard"
    }}

    If a field is numeric, convert it to a string. Authors must be strings in a list. If multiple authors, separate them into multiple items. The JSON must be strictly parseable with Python's json.loads().

    Text:
    \"\"\"
    {first_chunk_text}
    \"\"\"
    """

                # Call the OpenAI API
                response = openai.chat.completions.create(
                    model=self.PARSE_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant that extracts reference information from academic papers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                # Parse the assistant's reply
                assistant_reply = response.choices[0].message.content

                # Attempt to parse the assistant's reply as JSON
                references = json.loads(assistant_reply)

                # Update metadata
                data['references'].update(references)
                data['references_extracted'] = True
                self.save_metadata()

                print(f"Extracted references for {pdf_filename}.")
            except Exception as e:
                print(f"Error extracting references for {pdf_filename}: {e}")

    def generate_references_bib(self):
        """
        Generates the references.bib file using the data in the 'references' sub-dictionary
        of the metadata entries.
        """
        try:
            with open(self.REF_PATH, 'w', encoding='utf-8') as f:
                for pdf_filename, data in self.metadata.items():
                    references = data.get('references', {})
                    if not references:
                        continue  # Skip if no references data

                    # Use provided 'type' field; default to "misc" if not present
                    entry_type = references.get('type', 'misc')

                    # Use provided 'citation_key' field; if missing, generate a fallback from pdf_filename
                    citation_key = references.get('citation_key')
                    if not citation_key or not isinstance(citation_key, str) or citation_key.strip() == '':
                        # Fallback: all lowercase, remove special chars
                        base_key = os.path.splitext(pdf_filename)[0]
                        base_key = ''.join(ch for ch in base_key.lower() if ch.isalnum())
                        citation_key = base_key if base_key else 'unknownkey'

                    bib_entry = f"@{entry_type}{{{citation_key},\n"
                    for key, value in references.items():
                        # Skip 'type' and 'citation_key' fields as they are not BibTeX fields
                        if key in ['type', 'citation_key']:
                            continue

                        if value is None or value == '':
                            continue
                        # Convert value to string if it's not already
                        if isinstance(value, list):
                            # Convert all elements in the list to strings
                            value = [str(item) for item in value]
                            # Join list elements using ' and '
                            value = ' and '.join(value)
                        else:
                            value = str(value)

                        # Escape special characters in value
                        value = value.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')

                        bib_entry += f"  {key} = {{{value}}},\n"
                    bib_entry += "}\n\n"

                    f.write(bib_entry)
            print(f"Generated {self.REF_PATH} using references data.")
        except Exception as e:
            print(f"Error generating references.bib: {e}")

    def generate_embeddings(self, texts):
        """
        Generates embeddings for a list of texts.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: A list of embeddings.
        """
        response = openai.embeddings.create(
            input=texts,
            model=self.EMB_MODEL
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings

    def process_embeddings(self):
        """
        Generates embeddings for chunks listed in the metadata and updates their 'embedding_index'.
        Processes embeddings in batches for efficiency.
        """
        # Collect all chunks that need embeddings
        chunks_to_embed = []
        chunk_texts = []
        for pdf_filename, data in self.metadata.items():
            if not data.get('chunked'):
                print(f"Text for {pdf_filename} has not been chunked yet. Skipping embeddings.")
                continue

            # Go through each chunk
            chunks = data['chunks']
            for chunk_meta in chunks:
                if chunk_meta.get('embedding_index') is not None:
                    continue  # Embedding already exists
                # Read the chunk text
                chunk_filename = chunk_meta['filename']
                chunk_file_path = os.path.join(self.CHUNK_PATH, chunk_filename)
                try:
                    with open(chunk_file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    chunks_to_embed.append({
                        'pdf_filename': pdf_filename,
                        'chunk_meta': chunk_meta,
                        'text': text
                    })
                    chunk_texts.append(text)
                except Exception as e:
                    print(f"Error reading chunk {chunk_filename}: {e}")

        # Now, batch the embeddings
        if not chunks_to_embed:
            print("No new chunks to embed.")
            return

        total_chunks = len(chunks_to_embed)
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks_to_embed[i:i+self.batch_size]
            batch_texts = [item['text'] for item in batch]

            try:
                # Generate embeddings for the batch
                embeddings = self.generate_embeddings(batch_texts)
                # Append embeddings to self.embeddings
                start_index = len(self.embeddings)
                if len(self.embeddings) == 0:
                    self.embeddings = np.array(embeddings)
                else:
                    self.embeddings = np.vstack([self.embeddings, embeddings])
                # Update chunk_meta with embedding_index
                for idx, item in enumerate(batch):
                    embedding_index = start_index + idx
                    item['chunk_meta']['embedding_index'] = embedding_index
                    print(f"Generated embedding for chunk {item['chunk_meta']['filename']}.")
            except Exception as e:
                print(f"Error generating embeddings for batch starting at index {i}: {e}")
                continue

        # Save embeddings and metadata
        self.save_embeddings()
        self.save_metadata()
        print(f"Updated embeddings and metadata for all processed chunks.")

    def search(self, query, top_k=5):
        """
        Searches for the top_k chunks most similar to the query.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.

        Returns:
            List[dict]: List of dictionaries containing information about the top matching chunks.
        """
        # Generate embedding for the query
        query_embedding = self.generate_embeddings([query])[0]
        # Compute dot product between query embedding and embeddings
        embeddings = self.embeddings
        if len(embeddings) == 0:
            print("No embeddings available for search.")
            return []
        similarity_scores = np.dot(embeddings, query_embedding)
        # Get indices of top_k scores
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        # Map embedding indices back to chunks and PDFs
        results = []
        for idx in top_indices:
            # Find which chunk has this embedding index
            found = False
            for pdf_filename, data in self.metadata.items():
                for chunk_meta in data['chunks']:
                    if chunk_meta.get('embedding_index') == idx:
                        result = {
                            'pdf_filename': pdf_filename,
                            'chunk_filename': chunk_meta['filename'],
                            'similarity_score': float(similarity_scores[idx]),
                            'references': data.get('references', {})
                        }
                        results.append(result)
                        found = True
                        break  # Found the chunk
                if found:
                    break  # Found the PDF
        return results

if __name__ == "__main__":
    dm = DataManager()
    dm.update_metadata()
    dm.process_pdfs()
    dm.chunk_text_files()
    dm.process_embeddings()
