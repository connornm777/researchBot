import sys
import os
import json
import pyperclip
import time
import re
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QPushButton, QLabel, QFrame, QDialog,
                             QDialogButtonBox, QFileDialog, QScrollArea, QWidget, QSizePolicy, QShortcut)
from PyQt5.QtGui import QFont, QTextCursor, QKeyEvent, QKeySequence
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from dotenv import load_dotenv
import openai

from data_manager import DataManager  # Your DataManager file

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

GUI_PATH = os.getenv("GUI_PATH", ".")

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
        self.text_edit.moveCursor(QTextCursor.End)
        layout.addWidget(self.text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.setLayout(layout)
        self.resize(600, 400)
        self.snippet = snippet


class InputTextEdit(QTextEdit):
    """A QTextEdit that sends the message on Enter and inserts newline on Shift+Enter."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                # Shift+Enter -> newline
                self.insertPlainText("\n")
            else:
                # Enter -> send message
                self.parent().parent().send_message()
            return
        super().keyPressEvent(event)


class ReferenceChatGUI(QWidget):
    def __init__(self, dm):
        super().__init__()
        self.dm = dm

        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "When the user asks about references or citation keys, call "
                    "the 'search_references' function if needed. "
                    "If you use a snippet from the references I retrieved, please mark it with a numbered citation like [1], [2], etc. "
                    "Snippet numbering is cumulative throughout the entire conversation and never resets. "
                    "Snippet [1] is always the first snippet found in this session, snippet [2] the second, and so on. "
                    "If you reference snippet [1] again later, it still refers to that same original snippet. "
                    "Preserve these stable references even after saving/loading sessions."
                )
            }
        ]

        self.html_messages = []
        self.typing_indicator_visible = False

        # Keep track of all snippets found so far
        self.indexed_snippets = []
        # A cache for snippet stability: key=(pdf_filename, snippet), value=snippet_id
        self.snippet_cache = {}
        self.displayed_equations = []

        # Original send button stylesheet
        self.send_button_normal_style = "background-color: #4CAF50; color: white; padding: 6px 12px;"
        # Pressed/indented style
        self.send_button_pressed_style = "background-color: #4CAF50; color: white; padding: 6px 12px; margin-left: 10px;"

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("GPT Reference Assistant")

        # Dark mode stylesheet
        self.setStyleSheet("""
        QWidget {
            background-color: #202020;
            color: #ffffff;
        }
        QTextEdit {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        QPushButton {
            background-color: #4CAF50;
            border: 1px solid #555;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #55bd55;
        }
        QScrollArea {
            background-color: #202020;
        }
        QScrollBar:vertical {
            background: #2a2a2a;
            width: 12px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #4CAF50;
            border-radius: 4px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            background: none;
            height: 0;
        }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0,0,0,0)
        self.setLayout(main_layout)

        # Menu at top
        menu_layout = QHBoxLayout()
        new_button = QPushButton("New")
        new_button.clicked.connect(self.new_session)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_session)
        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_session)

        menu_layout.addWidget(new_button)
        menu_layout.addWidget(save_button)
        menu_layout.addWidget(load_button)
        menu_layout.addStretch()

        main_layout.addLayout(menu_layout)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(5,5,5,5)
        main_layout.addLayout(content_layout)

        # Left side: Chat + Input
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5,5,5,5)
        left_layout.setSpacing(5)

        title_label = QLabel("GPT Reference Assistant")
        title_font = QFont("Arial", 20, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignHCenter)
        title_label.setStyleSheet("color: #4CAF50;")
        left_layout.addWidget(title_label)

        self.web_view = QWebEngineView()
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.web_view)

        # Input frame at bottom
        input_frame = QFrame()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0,0,0,0)
        input_layout.setSpacing(5)
        input_frame.setLayout(input_layout)

        self.input_text = InputTextEdit()
        self.input_text.setFont(QFont("Arial", 14))
        self.input_text.setFixedHeight(100)

        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Arial", 14))
        self.send_button.setStyleSheet(self.send_button_normal_style)
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_text)
        input_layout.addWidget(self.send_button)

        left_layout.addWidget(input_frame, 0, Qt.AlignBottom)

        # Right side: citations panel
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(5,5,5,5)
        right_panel.setSpacing(5)
        self.right_panel_label = QLabel("Citations")
        self.right_panel_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.right_panel_label.setStyleSheet("color: #4CAF50;")
        right_panel.addWidget(self.right_panel_label)

        self.right_container = QFrame()
        self.right_inner_layout = QVBoxLayout()
        # Ensure small, fixed spacing
        self.right_inner_layout.setSpacing(0)
        self.right_inner_layout.setContentsMargins(5,5,5,5)
        self.right_container.setLayout(self.right_inner_layout)

        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.right_inner_layout)
        self.right_scroll.setWidget(scroll_widget)

        right_panel.addWidget(self.right_scroll)

        content_layout.addLayout(left_layout, stretch=2)
        content_layout.addLayout(right_panel, stretch=1)

        self.update_html()
        self.showMaximized()

        # Keyboard shortcuts
        self.shortcut_new = QShortcut(QKeySequence("Ctrl+N"), self)
        self.shortcut_new.activated.connect(self.new_session)

        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_session)

    def update_html(self):
        html_head = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {
  background-color: #202020;
  color: #ffffff;
  font-family: sans-serif;
  font-size: 16px;
  margin: 0px;
  padding: 20px;
  line-height: 1.4;
}
.message-user {
  background: #404040; 
  color: #ffffff;
  display: inline-block;
  padding: 10px;
  border-radius: 8px;
  margin-bottom: 6px;
  max-width: 60%;
}
.message-assistant {
  background: #303030; 
  color: #ffffff;
  display: inline-block;
  padding: 10px;
  border-radius: 8px;
  margin-bottom: 6px;
  max-width: 60%;
}
.message-container {
  text-align: left;
  margin: 6px 0;
}
.typing {
  color: #aaaaaa;
  font-style: italic;
  margin: 6px 0;
}
a {
  color: #4CAF50;
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
        html_body = "".join(self.html_messages)

        if self.typing_indicator_visible:
            html_body += '<div class="typing">GPT is typing...</div>'

        html_footer = """
</body>
</html>
"""
        full_html = html_head + html_body + html_footer
        self.web_view.setHtml(full_html)
        self.web_view.page().runJavaScript("MathJax.typeset();")
        self.web_view.page().runJavaScript("window.scrollTo(0, document.body.scrollHeight);")
        self.web_view.update()

    def append_user_message(self, message):
        self.html_messages.append(
            f'<div class="message-container"><div class="message-user"><b>You:</b><br>{message}</div></div>'
        )
        self.update_html()
        self.web_view.update()

    def append_assistant_message(self, message):
        if message and message.strip():
            self.html_messages.append(
                f'<div class="message-container"><div class="message-assistant"><b>GPT:</b><br>{message}</div></div>'
            )
            self.update_html()
            self.web_view.update()

    def show_typing_indicator(self, show=True):
        self.typing_indicator_visible = show
        self.update_html()
        self.web_view.update()

    def set_send_button_pressed(self, pressed=True):
        if pressed:
            self.send_button.setStyleSheet(self.send_button_pressed_style)
        else:
            self.send_button.setStyleSheet(self.send_button_normal_style)

    def send_message(self):
        user_input = self.input_text.toPlainText().strip()
        if not user_input:
            return

        self.append_user_message(user_input)
        self.input_text.clear()
        # Process events to ensure UI updates before API call
        QApplication.processEvents()

        # Add to history and show typing
        self.messages.append({"role": "user", "content": user_input})
        self.show_typing_indicator(True)

        # Indent send button while waiting
        self.set_send_button_pressed(True)

        # Now query GPT
        self.user_query(user_input)

    def user_query(self, user_message):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=self.messages,
            tools=tools
        )
        self.handle_response(response)

    def handle_response(self, response):
        self.show_typing_indicator(False)
        choice = response.choices[0]
        msg = choice.message

        tool_calls = getattr(msg, 'tool_calls', []) or []
        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "search_references":
                self.show_typing_indicator(True)
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
                # Unknown function
                self.set_send_button_pressed(False)
        else:
            assistant_content = msg.content if msg.content else ""
            if assistant_content.strip():
                self.messages.append({"role": "assistant", "content": assistant_content})
                self.append_assistant_message(assistant_content)
                eqs = self.extract_equations(assistant_content)
                for eq_full in eqs:
                    if eq_full not in self.displayed_equations:
                        self.displayed_equations.append(eq_full)
                self.update_right_panel()

            # Done responding
            self.set_send_button_pressed(False)

    def handle_final_answer(self, followup_response):
        choice = followup_response.choices[0]
        final_msg = choice.message
        assistant_content = final_msg.content if final_msg.content else ""
        if assistant_content.strip():
            self.messages.append({"role": "assistant", "content": assistant_content})
            self.append_assistant_message(assistant_content)
            eqs = self.extract_equations(assistant_content)
            for eq_full in eqs:
                if eq_full not in self.displayed_equations:
                    self.displayed_equations.append(eq_full)
        self.update_right_panel()

        # Done responding
        self.set_send_button_pressed(False)

    def handle_search_tool(self, query, top_k):
        results = self.dm.search(query, top_k=top_k)
        returned_data = []
        for r in results:
            snippet = self.get_snippet(r)
            citation_key = r['references'].get('citation_key', 'unknown_key')
            pdf_name = r['pdf_filename']

            # Check if we have seen this snippet before
            key = (pdf_name, snippet)
            if key in self.snippet_cache:
                snippet_id = self.snippet_cache[key]
            else:
                snippet_id = len(self.indexed_snippets) + 1
                self.indexed_snippets.append({
                    "id": snippet_id,
                    "citation_key": citation_key,
                    "pdf_name": pdf_name,
                    "snippet": snippet
                })
                self.snippet_cache[key] = snippet_id

            returned_data.append({
                "pdf_filename": pdf_name,
                "citation_key": citation_key,
                "snippet": snippet,
                "similarity_score": r['similarity_score']
            })
        return {"results": returned_data}

    def get_snippet(self, result):
        chunk_file = result['chunk_filename']
        chunk_path = os.path.join(self.dm.CHUNK_PATH, chunk_file)
        try:
            with open(chunk_path, 'r', encoding='utf-8') as f:
                snippet = f.read().strip()
        except:
            snippet = "Could not read chunk."
        return snippet

    def extract_equations(self, text):
        equations = []
        start = 0
        while True:
            start_eq = text.find("\\begin{equation}", start)
            if start_eq == -1:
                break
            end_eq = text.find("\\end{equation}", start_eq)
            if end_eq == -1:
                break
            eq_full = text[start_eq:end_eq+len("\\end{equation}")]
            equations.append(eq_full)
            start = end_eq+len("\\end{equation}")
        return equations

    def update_right_panel(self):
        # Clear old widgets
        for i in reversed(range(self.right_inner_layout.count())):
            widget = self.right_inner_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Display all snippets stacked with minimal spacing
        for s in self.indexed_snippets:
            result_frame = QFrame()
            result_layout = QHBoxLayout()
            result_layout.setSpacing(0)
            result_layout.setContentsMargins(0,0,0,0)
            result_frame.setLayout(result_layout)

            label_text = f"[{s['id']}] ({s['citation_key']})"
            snippet_label = QLabel(label_text)
            snippet_label.setFont(QFont("Arial", 12))
            snippet_label.setStyleSheet("color: #ffffff;")
            result_layout.addWidget(snippet_label)

            cite_button = QPushButton("Copy Cite")
            cite_button.setFont(QFont("Arial", 10))
            citation_text = f"\\cite{{{s['citation_key']}}}"
            cite_button.clicked.connect(lambda ch, ctext=citation_text: self.copy_to_clipboard(ctext))
            result_layout.addWidget(cite_button)

            snippet_button = QPushButton("Show Snippet")
            snippet_button.setFont(QFont("Arial", 10))
            snippet_button.clicked.connect(lambda ch, sn=s: self.show_snippet_dialog(sn['snippet'], sn['pdf_name']))
            result_layout.addWidget(snippet_button)

            self.right_inner_layout.addWidget(result_frame)

        # Equations directly after snippets
        for eq_full in self.displayed_equations:
            eq_frame = QFrame()
            eq_layout = QHBoxLayout()
            eq_layout.setSpacing(0)
            eq_layout.setContentsMargins(0,0,0,0)
            eq_frame.setLayout(eq_layout)

            eq_label = QLabel("[Equation]")
            eq_label.setFont(QFont("Arial", 12))
            eq_label.setStyleSheet("color: #ffffff;")
            eq_layout.addWidget(eq_label)

            copy_eq_button = QPushButton("Copy Equation")
            copy_eq_button.setFont(QFont("Arial", 10))
            copy_eq_button.clicked.connect(lambda ch, eq=eq_full: self.copy_to_clipboard(eq))
            eq_layout.addWidget(copy_eq_button)

            self.right_inner_layout.addWidget(eq_frame)

    def copy_to_clipboard(self, text):
        pyperclip.copy(text)

    def show_snippet_dialog(self, snippet, pdf_name):
        dialog = SnippetDialog(snippet, pdf_name, self)
        dialog.exec_()

    def new_session(self):
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "When the user asks about references or citation keys, call "
                    "the 'search_references' function if needed. "
                    "If you use a snippet from the references I retrieved, please mark it with a numbered citation like [1], [2], etc. "
                    "Snippet numbering is cumulative throughout the entire conversation and never resets. "
                    "Snippet [1] is always the first snippet found in this session, snippet [2] the second, and so on. "
                    "If you reference snippet [1] again later, it still refers to that same original snippet. "
                    "Preserve these stable references even after saving/loading sessions."
                )
            }
        ]
        self.html_messages = []
        self.indexed_snippets = []
        self.snippet_cache = {}
        self.displayed_equations = []
        self.typing_indicator_visible = False
        self.update_html()
        self.update_right_panel()
        self.set_send_button_pressed(False)
        print("New session started.")

    def save_session(self):
        session_data = {
            "messages": self.messages,
            "equations": self.displayed_equations,
            "snippets": self.indexed_snippets
        }
        timestamp = int(time.time())
        save_path = os.path.join(GUI_PATH, f"session_{timestamp}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=4)
        print(f"Session saved to {save_path}")

    def load_session(self):
        dlg = QFileDialog(self, "Load Session", GUI_PATH, "JSON Files (*.json)")
        dlg.setFileMode(QFileDialog.ExistingFile)
        if dlg.exec_() == QFileDialog.Accepted:
            file_path = dlg.selectedFiles()[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            self.messages = session_data.get("messages", [])
            self.displayed_equations = session_data.get("equations", [])
            self.indexed_snippets = session_data.get("snippets", [])

            # Rebuild the snippet_cache for stability
            self.snippet_cache = {}
            for s in self.indexed_snippets:
                key = (s['pdf_name'], s['snippet'])
                self.snippet_cache[key] = s['id']

            self.html_messages = []
            for m in self.messages:
                role = m.get('role', 'assistant')
                cont = m.get("content", None)
                if cont and cont.strip():
                    if role == 'user':
                        self.html_messages.append(
                            f'<div class="message-container"><div class="message-user"><b>You:</b><br>{cont}</div></div>'
                        )
                    elif role == 'assistant':
                        self.html_messages.append(
                            f'<div class="message-container"><div class="message-assistant"><b>GPT:</b><br>{cont}</div></div>'
                        )

            self.typing_indicator_visible = False
            self.update_html()
            self.update_right_panel()
            self.set_send_button_pressed(False)
            print(f"Session loaded from {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dm = DataManager()
    window = ReferenceChatGUI(dm)
    sys.exit(app.exec_())
