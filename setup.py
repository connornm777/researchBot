import os
import subprocess
from dotenv import load_dotenv

def main():
    # Load environment variables from .env
    load_dotenv()
    data_server = os.getenv('DATA_SERVER')
    if not data_server:
        raise ValueError("DATA_SERVER environment variable not set. Please read the README.md")

    # Paths as used by data_manager
    metadata_file = os.path.join(data_server, 'metadata.json')
    embedding_file = os.path.join(data_server, 'embeddings.npy')
    reference_file = os.path.join(data_server, 'references.bib')
    pdf_dir = os.path.join(data_server, 'pdfs')
    text_dir = os.path.join(data_server, 'text')
    chunks_dir = os.path.join(data_server, 'chunks')
    unscannable_dir = os.path.join(pdf_dir, 'unscannable')

    # Create directories if they don't exist
    os.makedirs(data_server, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(unscannable_dir, exist_ok=True)

    # Create placeholder files if they don't exist
    if not os.path.exists(metadata_file):
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("{}")
    # embeddings.npy and references.bib will be created by data_manager as needed,
    # so we don't need to create them now. If desired, we can create empty files:
    # if not os.path.exists(embedding_file):
    #     import numpy as np
    #     np.save(embedding_file, np.array([]))
    # if not os.path.exists(reference_file):
    #     with open(reference_file, 'w', encoding='utf-8') as f:
    #         f.write("")

    # Run data_manager.py to initialize and update metadata
    # This assumes data_manager.py is in the same directory as setup.py
    try:
        subprocess.run(["python", "data_manager.py"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error running data_manager.py:", e)
        return

    # Instructions for the user
    print("\nSetup complete.")
    print("You can now place your PDF files into the 'pdfs' directory located at:")
    print(pdf_dir)
    print("\nAfter adding or modifying PDFs, run 'data_manager.py' again to update the data.")
    print("For example:")
    print("    python data_manager.py\n")

if __name__ == "__main__":
    main()
