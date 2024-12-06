import sys
import os
import json
import re
import pyperclip  # For copying to clipboard
from dotenv import load_dotenv
import openai

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QPushButton, QLabel, QFrame, QDialog, QTextEdit,
                             QDialogButtonBox, QMenuBar, QMenu, QFileDialog, QMessageBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView

from data_manager import DataManager  # Your existing DataManager file

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
GUI_PATH = os.getenv("GUI_PATH", ".")

# Define the tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_references",
            "description": "Search the local corpus for relevant references and text chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the corpus."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return.",
                        "default": 3
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]

class SnippetDialog(QDialog):
    def __init__(self, snippet, pdf_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Snippet from {pdf_name}")
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(snippet)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        button_box = QHBoxLayout()
        copy_eq_button = QPushButton("Copy as Equation")
        copy_eq_button.clicked.connect(self.copy_as_equation)
        button_box.addWidget(copy_eq_button)

        ok_button = QPushButton("Ok")
        ok_button.clicked.connect(self.accept)
        button_box.addWidget(ok_button)

        box_frame = QFrame()
        box_frame.setLayout(button_box)

        layout.addWidget(box_frame)
        self.setLayout(layout)
        self.resize(600, 400)

        self.snippet = snippet

    def copy_as_equation(self):
        eq_text = f"\\begin{{equation}}\n{self.snippet}\n\\end{{equation}}"
        pyperclip.copy(eq_text)

class ReferenceChatGUI(QWidget):
    def __init__(self, dm):
        super().__init__()
        self.dm = dm

        # Messages in conversation
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "When the user asks about references or citation keys, call "
                    "the 'search_references' function if needed."
                )
            }
        ]

        # We will keep track of session data (messages, citations, equations)
        self.session_data = {
            "messages": self.messages,
            "citations": [],
            "equations": []
        }

        self.html_messages = []
        self.typing_indicator_visible = False
        self.last_search_results = []

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("GPT Reference Assistant")
        self.resize(1200, 800)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Menu bar
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("File")
        new_act = file_menu.addAction("New")
        new_act.triggered.connect(self.new_session)
        load_act = file_menu.addAction("Load")
        load_act.triggered.connect(self.load_session)
        save_act = file_menu.addAction("Save")
        save_act.triggered.connect(self.save_session)
        main_layout.setMenuBar(menubar)

        # Main content layout
        content_layout = QHBoxLayout()

        # Left side: Chat + Input
        left_layout = QVBoxLayout()

        title_label = QLabel("GPT Reference Assistant")
        title_font = QFont("Arial", 20, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignHCenter)
        left_layout.addWidget(title_label)

        self.web_view = QWebEngineView()
        left_layout.addWidget(self.web_view)

        input_frame = QFrame()
        input_layout = QHBoxLayout()
        input_frame.setLayout(input_layout)

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type your question here...")
        self.input_line.setFont(QFont("Arial", 14))
        self.input_line.returnPressed.connect(self.on_enter_pressed)
        send_button = QPushButton("Send")
        send_button.setFont(QFont("Arial", 14))
        send_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 6px 12px;")
        send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_line)
        input_layout.addWidget(send_button)
        left_layout.addWidget(input_frame)

        # Right side: citations and equations panel
        self.right_panel = QVBoxLayout()

        self.right_panel_label = QLabel("Relevant Citations and Equations:")
        self.right_panel_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.right_panel.addWidget(self.right_panel_label)

        self.right_container = QFrame()
        self.right_inner_layout = QVBoxLayout()
        self.right_container.setLayout(self.right_inner_layout)

        self.right_panel.addWidget(self.right_container)
        self.right_panel.addStretch()

        content_layout.addLayout(left_layout, stretch=3)
        content_layout.addLayout(self.right_panel, stretch=1)

        main_layout.addLayout(content_layout)
        self.update_html()

    def on_enter_pressed(self):
        self.send_message()

    def update_html(self):
        html_head = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {
  font-family: sans-serif;
  font-size: 16px;
  background: #f0f0f0;
  margin: 20px;
  line-height: 1.5;
}
.message-user {
  background: #e0e0e0; 
  color: #000000;
  display: inline-block;
  padding: 10px;
  border-radius: 8px;
  margin-bottom: 10px;
  max-width: 60%;
}
.message-assistant {
  background: #ffffff; 
  color: #000000;
  display: inline-block;
  padding: 10px;
  border-radius: 8px;
  margin-bottom: 10px;
  max-width: 60%;
}
.message-container {
  text-align: left;
  margin: 10px 0;
}
.typing {
  color: #999999;
  font-style: italic;
  margin: 10px 0;
}
</style>
<!-- Load MathJax -->
<script>
window.MathJax = {
  tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]}
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js" id="MathJax" async></script>
</head>
<body>
"""
        html_body = ""
        for msg_html in self.html_messages:
            html_body += msg_html

        if self.typing_indicator_visible:
            html_body += '<div class="typing">GPT is typing...</div>'

        html_footer = """
</body>
</html>
"""
        full_html = html_head + html_body + html_footer
        self.web_view.setHtml(full_html)
        # After rendering, typeset math and scroll down
        self.web_view.page().runJavaScript("MathJax.typeset();", self.scroll_down)

    def scroll_down(self, _=None):
        self.web_view.page().runJavaScript("window.scrollTo(0, document.body.scrollHeight);")

    def append_user_message(self, message):
        self.html_messages.append(
            f'<div class="message-container"><div class="message-user"><b>You:</b><br>{message}</div></div>'
        )
        self.update_html()

    def append_assistant_message(self, message):
        self.html_messages.append(
            f'<div class="message-container"><div class="message-assistant"><b>GPT:</b><br>{message}</div></div>'
        )
        self.update_html()

    def show_typing_indicator(self, show=True):
        self.typing_indicator_visible = show
        self.update_html()

    def send_message(self):
        user_input = self.input_line.text().strip()
        if not user_input:
            return

        # Render user's message before adding to history or querying GPT
        self.append_user_message(user_input)
        self.input_line.clear()
        QApplication.processEvents()

        # Now add to history and call GPT
        self.session_data["messages"].append({"role": "user", "content": user_input})
        self.show_typing_indicator(True)
        QApplication.processEvents()

        self.user_query(user_input)

    def user_query(self, user_message):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=self.session_data["messages"],
            tools=tools
        )
        self.handle_response(response)

    def handle_response(self, response):
        choice = response.choices[0]
        msg = choice.message

        tool_calls = getattr(msg, 'tool_calls', []) or []
        self.show_typing_indicator(False)
        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "search_references":
                self.show_typing_indicator(True)
                QApplication.processEvents()

                results = self.handle_search_tool(arguments["query"], arguments.get("top_k", 3))
                tool_response_message = {
                    "role": "tool",
                    "content": json.dumps(results),
                    "tool_call_id": tool_call.id
                }
                self.session_data["messages"].append(msg.dict())
                self.session_data["messages"].append(tool_response_message)

                followup_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=self.session_data["messages"],
                    tools=tools
                )
                self.show_typing_indicator(False)
                self.handle_final_answer(followup_response)
            else:
                self.append_assistant_message("I tried to call an unknown function.")
        else:
            assistant_content = msg.content
            if assistant_content:
                self.session_data["messages"].append({"role": "assistant", "content": assistant_content})
                self.append_assistant_message(assistant_content)
                # Update right panel (no new tool call)
                self.extract_equations(assistant_content)
                self.update_right_panel()

    def handle_final_answer(self, followup_response):
        choice = followup_response.choices[0]
        final_msg = choice.message
        assistant_content = final_msg.content
        if assistant_content:
            self.session_data["messages"].append({"role": "assistant", "content": assistant_content})
            self.append_assistant_message(assistant_content)
        self.extract_equations(assistant_content)
        self.update_right_panel()

    def handle_search_tool(self, query, top_k):
        results = self.dm.search(query, top_k=top_k)
        data = []
        self.last_search_results = results  # Store for right panel
        # Add citations to session_data
        for r in results:
            pdf_name = r['pdf_filename']
            citation_key = r['references'].get('citation_key', 'unknown_key')
            chunk_file = r['chunk_filename']
            chunk_path = os.path.join(self.dm.CHUNK_PATH, chunk_file)
            try:
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    snippet = f.read().strip()
            except:
                snippet = "Could not read chunk."
            r['snippet'] = snippet
            r['pdf_filename'] = pdf_name
            r['citation_key'] = citation_key

            data.append({
                "pdf_filename": pdf_name,
                "citation_key": citation_key,
                "snippet": snippet,
                "similarity_score": r['similarity_score']
            })
            # Store citation keys
            self.session_data["citations"].append(citation_key)
        return {"results": data}

    def extract_equations(self, text):
        # Find all equations of the form:
        # \begin{equation}
        # ...
        # \end{equation}
        pattern = r'\\begin{equation}(.*?)\\end{equation}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for eq in matches:
            eq_str = eq.strip()
            self.session_data["equations"].append(eq_str)

    def update_right_panel(self):
        # Clear old widgets
        for i in reversed(range(self.right_inner_layout.count())):
            widget = self.right_inner_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # First show citations if any
        if self.last_search_results:
            # Citations from last search
            for r in self.last_search_results:
                citation_key = r['citation_key']
                snippet = r['snippet']
                pdf_name = r['pdf_filename']

                result_frame = QFrame()
                result_layout = QHBoxLayout()
                result_frame.setLayout(result_layout)

                cite_key_label = QLabel(f"Citation: {citation_key}")
                cite_key_label.setFont(QFont("Arial", 12))
                result_layout.addWidget(cite_key_label)

                # Button to copy citation
                cite_button = QPushButton("Copy Cite")
                cite_button.setFont(QFont("Arial", 10))
                citation_text = f"\\cite{{{citation_key}}}"
                cite_button.clicked.connect(lambda ch, ctext=citation_text: self.copy_to_clipboard(ctext))
                result_layout.addWidget(cite_button)

                # Button to show snippet
                snippet_button = QPushButton("Show Snippet")
                snippet_button.setFont(QFont("Arial", 10))
                snippet_button.clicked.connect(lambda ch, s=snippet, p=pdf_name: self.show_snippet_dialog(s, p))
                result_layout.addWidget(snippet_button)

                self.right_inner_layout.addWidget(result_frame)

        # Now show equations from the entire session
        if self.session_data["equations"]:
            eq_label = QLabel("Equations:")
            eq_label.setFont(QFont("Arial", 14, QFont.Bold))
            self.right_inner_layout.addWidget(eq_label)

            for eq_str in self.session_data["equations"]:
                eq_frame = QFrame()
                eq_layout = QHBoxLayout()
                eq_frame.setLayout(eq_layout)

                eq_preview = QLabel("Equation found")
                eq_preview.setFont(QFont("Arial", 12))
                eq_layout.addWidget(eq_preview)

                copy_eq_button = QPushButton("Copy Equation")
                copy_eq_button.setFont(QFont("Arial", 10))
                full_eq = f"\\begin{{equation}}\n{eq_str}\n\\end{{equation}}"
                copy_eq_button.clicked.connect(lambda ch, feq=full_eq: self.copy_to_clipboard(feq))
                eq_layout.addWidget(copy_eq_button)

                self.right_inner_layout.addWidget(eq_frame)

    def copy_to_clipboard(self, text):
        pyperclip.copy(text)

    def show_snippet_dialog(self, snippet, pdf_name):
        dialog = SnippetDialog(snippet, pdf_name, self)
        dialog.exec_()

    def new_session(self):
        reply = QMessageBox.question(
            self, "New Session", "Are you sure you want to start a new session? Unsaved data will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful research assistant. "
                        "When the user asks about references or citation keys, call "
                        "the 'search_references' function if needed."
                    )
                }
            ]
            self.session_data = {
                "messages": self.messages,
                "citations": [],
                "equations": []
            }
            self.html_messages = []
            self.last_search_results = []
            self.update_right_panel()
            self.update_html()

    def save_session(self):
        if not os.path.exists(GUI_PATH):
            os.makedirs(GUI_PATH)
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Session", GUI_PATH, "JSON Files (*.json)")
        if file_path:
            # Save session_data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=4)
            QMessageBox.information(self, "Saved", f"Session saved to {file_path}")

    def load_session(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Session", GUI_PATH, "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)

            # Load back into UI
            self.session_data = loaded
            self.messages = self.session_data["messages"]

            # Rebuild the html_messages from messages
            self.html_messages = []
            for m in self.messages:
                role = m["role"]
                content = m["content"]
                if role == "user":
                    self.html_messages.append(
                        f'<div class="message-container"><div class="message-user"><b>You:</b><br>{content}</div></div>'
                    )
                elif role == "assistant":
                    self.html_messages.append(
                        f'<div class="message-container"><div class="message-assistant"><b>GPT:</b><br>{content}</div></div>'
                    )
            self.last_search_results = []  # no stored last search results
            # We only have citations and equations
            self.update_right_panel()
            self.update_html()
            QMessageBox.information(self, "Loaded", f"Session loaded from {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dm = DataManager()
    window = ReferenceChatGUI(dm)
    window.show()
    sys.exit(app.exec_())
