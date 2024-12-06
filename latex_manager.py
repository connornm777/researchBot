import os
import json


class LatexProjectManager:
    def __init__(self, project_path):
        """
        Initialize with a directory containing .tex files.
        """
        self.project_path = project_path
        self.files = {}  # {filename: {"path": full_path, "content": "text of file"}}
        self.load_files()

    def load_files(self):
        """
        Load all .tex files in the project directory into memory.
        """
        for fname in os.listdir(self.project_path):
            if fname.endswith(".tex"):
                full_path = os.path.join(self.project_path, fname)
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.files[fname] = {
                    "path": full_path,
                    "content": content
                }

    def save_file(self, fname):
        """
        Save a file's content back to disk.
        """
        if fname in self.files:
            full_path = self.files[fname]["path"]
            content = self.files[fname]["content"]
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

    def insert_citation(self, fname, citation_key, line_number=None, search_pattern=None):
        """
        Insert a citation into a file. Either at a given line number, or find a search pattern
        (like a particular section or figure reference) and insert after that.
        """
        if fname not in self.files:
            return False, "File not found."
        lines = self.files[fname]["content"].split('\n')

        if line_number is not None:
            # Insert \cite{citation_key} at the given line_number
            if line_number < 0 or line_number >= len(lines):
                return False, "Invalid line number."
            # Append citation at the end of the line
            lines[line_number] += f" \\cite{{{citation_key}}}"
        elif search_pattern is not None:
            # Find the line containing search_pattern and insert citation
            found = False
            for i, line in enumerate(lines):
                if search_pattern in line:
                    lines[i] += f" \\cite{{{citation_key}}}"
                    found = True
                    break
            if not found:
                return False, "Pattern not found."
        else:
            # Default: append at the end of the file
            lines.append(f"\\cite{{{citation_key}}}")

        self.files[fname]["content"] = '\n'.join(lines)
        self.save_file(fname)
        return True, "Citation inserted."

    def get_file_content(self, fname):
        """
        Return the content of a .tex file.
        """
        if fname in self.files:
            return self.files[fname]["content"]
        return None

    def list_files(self):
        """
        List all managed .tex files.
        """
        return list(self.files.keys())

    def ask_gpt_for_help(self, user_query, openai_api_key):
        """
        Ask GPT for suggestions on editing LaTeX files.

        This could be integrated with a tool definition, where GPT might call a function like:
        - "insert_citation"
        - "suggest_improvements"

        For now, we'll just do a basic chat completion:
        """
        import openai
        openai.api_key = openai_api_key
        messages = [
            {"role": "system", "content": "You are a helpful LaTeX editing assistant."},
            {"role": "user", "content": user_query}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content

    def suggest_improvements(self, fname, openai_api_key):
        """
        Ask GPT to suggest improvements to a given LaTeX file.
        """
        content = self.get_file_content(fname)
        if content is None:
            return "File not found."
        query = f"Here is a LaTeX file content. Suggest improvements or formatting tips:\n\n{content}"
        return self.ask_gpt_for_help(query, openai_api_key)
