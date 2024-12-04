import os
import json
import openai
from dotenv import load_dotenv, dotenv_values
import numpy as np
from pdfminer.high_level import extract_text
from PyPDF2 import PdfReader

class DataManager:
    def __init__(self):

        for key, value in dotenv_values().items():
            setattr(self, key, value)

        openai.api_key = getattr(self, 'OPENAI_API_KEY', None)
        self.metadata = self.load_metadata()
        self.embeddings = self.load_embeddings()

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
            return []

    def save_embeddings(self):
        np.save(self.EMB_PATH, self.embeddings)

    def convert_pdf_to_text(self, pdf_filename):
        pdf_path = os.path.join(self.PDF_PATH, pdf_filename)
        text_filename = os.path.splitext(pdf_filename)[0] + '.txt'
        text_path = os.path.join(self.TXT_PATH, text_filename)

        os.makedirs(self.TXT_PATH, exist_ok=True)

        text = extract_text(pdf_path)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return text_filename  # Return the name of the text file created

    def chunk_text_file(self, text_filename, chunk_size=500):
        text_path = os.path.join(self.TXT_PATH, text_filename)
        base_filename = os.path.splitext(text_filename)[0]
        os.makedirs(self.CHUNK_PATH, exist_ok=True)

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunk_filename = f"{base_filename}_chunk{i // chunk_size}.txt"
            chunk_file_path = os.path.join(self.CHUNK_PATH, chunk_filename)
            with open(chunk_file_path, 'w', encoding='utf-8') as f:
                f.write(chunk_text)
            chunks.append(chunk_filename)
        return chunks  # Return list of chunk filenames

    def generate_embedding(self, text):
        response = openai.Embedding.create(
            input=text,
            model='text-embedding-ada-002'
        )
        embedding = response['data'][0]['embedding']
        return embedding

    def generate_embeddings_for_chunks(self, chunk_filenames):
        new_embeddings = []
        for chunk_filename in chunk_filenames:
            if chunk_filename in self.embeddings['files']:
                print(f"Embedding for {chunk_filename} already exists. Skipping.")
                continue
            chunk_file_path = os.path.join(self.CHUNK_PATH, chunk_filename)
            with open(chunk_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            embedding = self.generate_embedding(text)
            new_embeddings.append(embedding)
            self.embeddings['files'].append(chunk_filename)
            print(f"Generated embedding for {chunk_filename}.")
        if new_embeddings:
            # Append new embeddings to existing array
            if self.embeddings['embeddings'].size == 0:
                self.embeddings['embeddings'] = np.array(new_embeddings)
            else:
                self.embeddings['embeddings'] = np.vstack((self.embeddings['embeddings'], new_embeddings))
            self.save_embeddings()

    def extract_pdf_metadata(self, pdf_filename):
        pdf_path = os.path.join(self.PDF_PATH, pdf_filename)
        reader = PdfReader(pdf_path)
        info = reader.metadata
        metadata = {
            'Title': info.title if info.title else '',
            'Authors': []
        }
        # Extract authors
        if info.author:
            authors = info.author.replace(';', ',').split(',')
            metadata['Authors'] = [author.strip() for author in authors]
        else:
            metadata['Authors'] = []

        metadata['Affiliations'] = []  # Additional parsing can be added
        return metadata


