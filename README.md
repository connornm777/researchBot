# Wrapper for GPT API

## Setup

1) Rename .env.example to .env
2) Edit .env to put in your access token (OPENAI_API_KEY) and path to store your data (DATA_SERVER)
3) Run setup.py 
4) Put text-scannable pdfs into the pdfs directory in (DATA_SERVER)
5) Run data_manager.py to process the files so gui_chat.py can access them
6) Run guit_chat.py, and interact with your pdfs!

A references.bib is generated from all of these. 