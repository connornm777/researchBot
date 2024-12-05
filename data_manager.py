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
                    'embedded': False,
                    'metadata_extracted': False,
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
        fields_to_remove = ['title', 'authors', 'affiliations', 'embedded']
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

    def extract_references(self):
        """
        Uses GPT to extract reference information from the first chunk
        of each text file, and updates the metadata under the 'references' field.
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
                prompt = f"""
Extract any reference information from the text below. Possible fields include, but are not limited to:

- Title of the paper
- Authors
- Affiliations
- Journal
- Year
- Volume
- Issue
- Pages
- DOI
- URL

If any information is missing, simply omit that field.

Return the information in JSON format.

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
                        {"role": "system", "content": "You are a helpful assistant that extracts reference information from academic papers."},
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

                    # Use the PDF filename without extension as the citation key
                    citation_key = os.path.splitext(pdf_filename)[0]

                    # Build the BibTeX entry dynamically based on available fields
                    bib_entry = f"@article{{{citation_key},\n"
                    for key, value in references.items():
                        # Convert key to string if it's not already
                        if not isinstance(key, str):
                            key = str(key)
                        # Skip keys with empty or None values
                        if value is None or value == '':
                            continue
                        # Convert value to string if it's not already
                        if isinstance(value, list):
                            # Convert all elements in the list to strings
                            value = [str(item) for item in value]
                            # Join list elements using ' and ' as per BibTeX format for multiple authors
                            value = ' and '.join(value)
                        else:
                            # Convert value to string
                            value = str(value)
                        # Escape special characters in value
                        value = value.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
                        # Add the field to the BibTeX entry
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
