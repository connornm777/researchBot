import sys
import os
import json
import pyperclip  # For copying to clipboard (pip install pyperclip)
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QPushButton, QLabel, QFrame, QDialog, QTextEdit, QDialogButtonBox)
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from dotenv import load_dotenv
import openai

from data_manager import DataManager  # Your existing DataManager file

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    def __init__(self, snippet, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Snippet")
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText(snippet)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.resize(600, 400)


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

        # Store the conversation as HTML
        self.html_messages = []
        self.typing_indicator_visible = False

        # Will store search results from the last tool call to display on the right panel
        self.last_search_results = []

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("GPT Reference Assistant")
        self.resize(1200, 800)

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

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

        # Right side: citations and chunks panel
        self.right_panel = QVBoxLayout()
        self.right_panel_label = QLabel("Relevant Citations and Chunks:")
        self.right_panel_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.right_panel.addWidget(self.right_panel_label)

        self.right_container = QFrame()
        self.right_inner_layout = QVBoxLayout()
        self.right_container.setLayout(self.right_inner_layout)

        self.right_panel.addWidget(self.right_container)
        self.right_panel.addStretch()

        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(self.right_panel, stretch=1)

        self.update_html()

        self.show()

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
        self.web_view.page().runJavaScript("MathJax.typeset();")

    def append_user_message(self, message):
        # Both on left side, different bubble color
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

        # Render the user's message before adding to history or querying GPT
        self.append_user_message(user_input)
        self.input_line.clear()
        QApplication.processEvents()

        # Now that it's displayed, we add to history and call GPT
        self.messages.append({"role": "user", "content": user_input})
        self.show_typing_indicator(True)
        QApplication.processEvents()

        self.user_query(user_input)

    def user_query(self, user_message):
        response = openai.chat.completions.create(
            model="gpt-4o",  # Use a model supporting these features
            messages=self.messages,
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
                self.messages.append(msg.dict())
                self.messages.append(tool_response_message)

                followup_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=self.messages,
                    tools=tools
                )
                self.show_typing_indicator(False)
                self.handle_final_answer(followup_response)
            else:
                self.append_assistant_message("I tried to call an unknown function.")
        else:
            assistant_content = msg.content
            if assistant_content:
                self.messages.append({"role": "assistant", "content": assistant_content})
                self.append_assistant_message(assistant_content)
                # Update right panel if needed (no tool call results means no new citations)
                self.update_right_panel()

    def handle_final_answer(self, followup_response):
        choice = followup_response.choices[0]
        final_msg = choice.message
        assistant_content = final_msg.content
        if assistant_content:
            self.messages.append({"role": "assistant", "content": assistant_content})
            self.append_assistant_message(assistant_content)
        # Update right panel after final answer
        self.update_right_panel()

    def handle_search_tool(self, query, top_k):
        results = self.dm.search(query, top_k=top_k)
        data = []
        self.last_search_results = results  # Store to show on right panel
        for r in results:
            pdf_name = r['pdf_filename']
            citation_key = r['references'].get('citation_key', 'unknown_key')
            snippet = r.get('snippet', '')
            # Actually snippet is read inside the handle_search_tool in original code.
            # We'll get it from the chunk_file again here:
            chunk_file = r['chunk_filename']
            chunk_path = os.path.join(self.dm.CHUNK_PATH, chunk_file)
            try:
                with open(chunk_path, 'r', encoding='utf-8') as f:
                    snippet = f.read().strip()
            except:
                snippet = "Could not read chunk."
            r['snippet'] = snippet

            data.append({
                "pdf_filename": pdf_name,
                "citation_key": citation_key,
                "snippet": snippet,
                "similarity_score": r['similarity_score']
            })
        return {"results": data}

    def update_right_panel(self):
        # Clear old widgets
        for i in reversed(range(self.right_inner_layout.count())):
            widget = self.right_inner_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Display the citations and snippets from last search results if any
        if not self.last_search_results:
            # If no search results, maybe just show nothing
            no_result_label = QLabel("No recent citations.")
            self.right_inner_layout.addWidget(no_result_label)
        else:
            for r in self.last_search_results:
                citation_key = r['references'].get('citation_key', 'unknown_key')
                pdf_name = r['pdf_filename']
                snippet = r['snippet']

                # A frame for each result
                result_frame = QFrame()
                result_layout = QHBoxLayout()
                result_frame.setLayout(result_layout)

                # Label with PDF name
                pdf_label = QLabel(pdf_name)
                pdf_label.setFont(QFont("Arial", 12))
                result_layout.addWidget(pdf_label)

                # Button to copy citation
                cite_button = QPushButton("Copy Cite")
                cite_button.setFont(QFont("Arial", 10))
                citation_text = f"\\cite{{{citation_key}}}"
                cite_button.clicked.connect(lambda ch, ctext=citation_text: self.copy_to_clipboard(ctext))
                result_layout.addWidget(cite_button)

                # Button to show snippet
                snippet_button = QPushButton("Show Snippet")
                snippet_button.setFont(QFont("Arial", 10))
                snippet_button.clicked.connect(lambda ch, s=snippet: self.show_snippet_dialog(s))
                result_layout.addWidget(snippet_button)

                self.right_inner_layout.addWidget(result_frame)

    def copy_to_clipboard(self, text):
        pyperclip.copy(text)

    def show_snippet_dialog(self, snippet):
        dialog = SnippetDialog(snippet, self)
        dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dm = DataManager()
    window = ReferenceChatGUI(dm)
    sys.exit(app.exec_())
