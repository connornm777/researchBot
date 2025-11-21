import sys
import os
import json
import time
import re
import threading
import subprocess
from typing import Optional, List, Dict, Any

import pyperclip
from dotenv import load_dotenv

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import (
    QFont,
    QTextCursor,
    QKeyEvent,
    QKeySequence,
    QDesktopServices,
)
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QFrame,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QScrollArea,
    QSizePolicy,
    QShortcut,
    QCheckBox,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView

from openai import OpenAI

from data_manager import DataManager, run_pipeline


# ---------- config ----------

load_dotenv()


def _env(primary: str, *alts: str, default: Optional[str] = None) -> Optional[str]:
    """Return first non-empty env var among [primary, *alts], else default."""
    for k in (primary, *alts):
        v = os.getenv(k)
        if v:
            return v
    return default


CHAT_MODEL = _env("CHAT_MODEL", default="gpt-4o-mini")
DATA_SERVER = _env("DATA_SERVER", default=".")
APP_SESSIONS_PATH = os.path.join(DATA_SERVER, "app_sessions")
os.makedirs(APP_SESSIONS_PATH, exist_ok=True)

THESIS_DIR = os.path.join(DATA_SERVER, "thesis")
os.makedirs(THESIS_DIR, exist_ok=True)

MAX_HISTORY = 40  # system + last 39 messages
DEFAULT_TOP_K = 5

# ---- tools exposed to the model (for thesis operations) ----

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_thesis_metadata",
            "description": (
                "Return a list of sections and subsections in the currently loaded thesis file. "
                "Use this to discover what parts of the thesis exist before editing."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_thesis_section",
            "description": (
                "Return the LaTeX source of a specific section or subsection of the thesis. "
                "Use the 'index' value from read_thesis_metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Index of the section/subsection from read_thesis_metadata (0-based).",
                    }
                },
                "required": ["index"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_thesis_section",
            "description": (
                "Replace the LaTeX source of a specific section or subsection of the thesis with new text. "
                "Call this only after you and the user agree on the final version."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Index of the section/subsection from read_thesis_metadata (0-based).",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Full replacement LaTeX source for that section, including its \\section/\\subsection line.",
                    },
                },
                "required": ["index", "new_text"],
                "additionalProperties": False,
            },
        },
    },
]



client = OpenAI()


def escape_html(text: str) -> str:
    """Basic HTML escaping with newline -> <br>."""
    if text is None:
        return ""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\n", "<br>")
    return text


class SnippetDialog(QDialog):
    def __init__(self, snippet: str, pdf_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Snippet from {pdf_name}")
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit(self)
        self.text_edit.setPlainText(snippet)
        self.text_edit.setReadOnly(True)
        self.text_edit.moveCursor(QTextCursor.End)
        layout.addWidget(self.text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok, parent=self)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class InputTextEdit(QTextEdit):
    """
    QTextEdit that:
      - Enter        -> submit
      - Shift+Enter  -> newline
    """

    submitted = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                self.insertPlainText("\n")
            else:
                self.submitted.emit()
            return
        super().keyPressEvent(event)


class ReferenceChatGUI(QWidget):
    # signals for background model calls
    model_reply_ready = pyqtSignal(str)
    model_error = pyqtSignal(str)

    def __init__(self, dm: DataManager):
        super().__init__()
        self.dm = dm

        # conversation state
        self.messages: List[Dict[str, Any]] = []
        self.html_messages: List[str] = []
        self.typing_indicator_visible = False

        # RAG snippet state
        self.indexed_snippets: List[Dict[str, Any]] = []
        self.snippet_cache: Dict[tuple, int] = {}  # (pdf_filename, snippet) -> id
        self.displayed_equations: List[str] = []

        # settings
        self.citations_enabled = True

        # TeX / thesis state
        self.current_tex_path: Optional[str] = None
        self.current_tex_content: str = ""
        self.thesis_sections: List[Dict[str, Any]] = []
        self.current_thesis_section: Optional[Dict[str, Any]] = None
        self.current_thesis_section_text: str = ""

        self.send_button_normal_style = (
            "background-color: #4CAF50; color: white; padding: 6px 12px;"
        )
        self.send_button_pressed_style = (
            "background-color: #4CAF50; color: white; padding: 6px 12px; margin-left: 10px;"
        )

        # connect signals
        self.model_reply_ready.connect(self.on_model_reply_ready)
        self.model_error.connect(self.on_model_error)

        self.init_ui()
        self.new_session()

    # ---------- UI ----------


    def init_ui(self):
        self.setWindowTitle("GPT Reference Assistant")

        self.setStyleSheet(
            """
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
            border-radius: 4px;
            padding: 4px 8px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QScrollArea {
            border: none;
        }
        QScrollBar:vertical {
            background: #2a2a2a;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #555555;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            background: none;
            height: 0;
        }
        """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # top menu
        menu_layout = QHBoxLayout()

        new_button = QPushButton("New")
        new_button.clicked.connect(self.new_session)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_session)

        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_session)

        self.citations_checkbox = QCheckBox("Enable citations")
        self.citations_checkbox.setChecked(True)
        self.citations_checkbox.stateChanged.connect(
            lambda state: setattr(self, "citations_enabled", bool(state))
        )

        thesis_load_button = QPushButton("Load Thesis")
        thesis_load_button.clicked.connect(self.load_thesis_from_default)

        thesis_compile_button = QPushButton("Compile PDF")
        thesis_compile_button.clicked.connect(self.compile_thesis)

        thesis_pdf_button = QPushButton("Open PDF")
        thesis_pdf_button.clicked.connect(self.open_thesis_pdf)

        menu_layout.addWidget(new_button)
        menu_layout.addWidget(save_button)
        menu_layout.addWidget(load_button)
        menu_layout.addWidget(self.citations_checkbox)
        menu_layout.addStretch()
        menu_layout.addWidget(thesis_load_button)
        menu_layout.addWidget(thesis_compile_button)
        menu_layout.addWidget(thesis_pdf_button)
        main_layout.addLayout(menu_layout)

        # main split
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addLayout(content_layout)

        # left: chat
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)

        title_label = QLabel("GPT Reference Assistant")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignHCenter)
        title_label.setStyleSheet("color: #4CAF50;")
        left_layout.addWidget(title_label)

        self.web_view = QWebEngineView(self)
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.web_view)

        # input
        input_frame = QFrame(self)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(5)

        self.input_text = InputTextEdit(self)
        self.input_text.setFont(QFont("Arial", 14))
        self.input_text.setFixedHeight(90)
        self.input_text.submitted.connect(self.send_message)

        self.send_button = QPushButton("Send", self)
        self.send_button.setFont(QFont("Arial", 14))
        self.send_button.setStyleSheet(self.send_button_normal_style)
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_text)
        input_layout.addWidget(self.send_button)
        left_layout.addWidget(input_frame)

        # right: snippets & equations
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)

        self.right_panel_label = QLabel("Citations & Equations")
        self.right_panel_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.right_panel_label.setStyleSheet("color: #4CAF50;")
        right_layout.addWidget(self.right_panel_label)

        self.right_scroll = QScrollArea(self)
        self.right_scroll.setWidgetResizable(True)
        self.right_inner_layout = QVBoxLayout()
        self.right_inner_layout.setContentsMargins(5, 5, 5, 5)
        self.right_inner_layout.setSpacing(5)
        right_widget = QWidget()
        right_widget.setLayout(self.right_inner_layout)
        self.right_scroll.setWidget(right_widget)
        right_layout.addWidget(self.right_scroll)

        content_layout.addLayout(left_layout, stretch=2)
        content_layout.addLayout(right_layout, stretch=1)

        # shortcuts
        self.shortcut_new = QShortcut(QKeySequence("Ctrl+N"), self)
        self.shortcut_new.activated.connect(self.new_session)
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_session)
        self.shortcut_load = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_load.activated.connect(self.load_session)

        self.update_html()
        self.showMaximized()

    # ---------- chat helpers ----------

    def get_limited_history(self) -> List[Dict[str, Any]]:
        msgs = self.messages
        if len(msgs) <= MAX_HISTORY:
            return msgs
        return [msgs[0]] + msgs[-(MAX_HISTORY - 1):]

    def show_typing_indicator(self, show: bool = True):
        self.typing_indicator_visible = show
        self.update_html()

    def set_send_button_pressed(self, pressed: bool = True):
        self.send_button.setEnabled(not pressed)
        if pressed:
            self.send_button.setStyleSheet(self.send_button_pressed_style)
        else:
            self.send_button.setStyleSheet(self.send_button_normal_style)

    def append_user_message(self, message_html: str):
        self.html_messages.append(
            f'<div class="message-container"><div class="message-user"><b>You:</b><br>{message_html}</div></div>'
        )
        self.update_html()

    def append_assistant_message(self, message_html: str):
        if not message_html or not message_html.strip():
            return
        self.html_messages.append(
            f'<div class="message-container"><div class="message-assistant"><b>GPT:</b><br>{message_html}</div></div>'
        )
        self.update_html()

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
  margin: 0;
  padding: 20px;
  line-height: 1.4;
}
.message-container {
  margin-bottom: 10px;
}
.message-user {
  background-color: #303030;
  border-radius: 8px;
  padding: 8px 10px;
}
.message-assistant {
  background-color: #252525;
  border-radius: 8px;
  padding: 8px 10px;
}
.typing {
  color: #aaaaaa;
  font-style: italic;
  margin: 6px 0;
}
a { color: #4CAF50; }
</style>
<script>
window.MathJax = {
  tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]}
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"
        id="MathJax" async></script>
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
        self.web_view.setHtml(html_head + html_body + html_footer)

    # ---------- session ----------

    def new_session(self):
        self.messages = [
            {
                "role": "system",
                "content": (
                    "- You are a research assistant helping with a LaTeX-based physics thesis.\n"
                    "- The user maintains a local PDF corpus; you may be given snippets from it as context.\n"
                    "- The user also has a main LaTeX thesis file; when a 'current section' is provided, "
                    "treat that section as primary context and edit only that section unless instructed otherwise.\n"
                    "- DO NOT narrate internal reasoning, searches, or tools. Just provide final answers.\n"
                    "- When citing, always use LaTeX \\cite{<citation_key>} with keys that are explicitly given "
                    "in the context or by the user. Never invent new citation keys.\n"
                    "- Preserve LaTeX structure and commands; only adjust wording / insert citations.\n"
             ),
            }
        ]
        self.html_messages = []
        self.indexed_snippets = []
        self.snippet_cache = {}
        self.displayed_equations = []
        self.current_thesis_section = None
        self.current_thesis_section_text = ""
        self.typing_indicator_visible = False

        self.update_html()
        self.update_right_panel()
        self.set_send_button_pressed(False)
        print("New session started.")

    # ---------- sending / commands ----------

    def send_message(self):
        raw_text = self.input_text.toPlainText()
        stripped = raw_text.strip()
        if not stripped:
            return

        # show user message immediately
        self.append_user_message(escape_html(raw_text))
        self.input_text.clear()

        # command mode
        if stripped.startswith(":"):
            handled = self.handle_command(stripped)
            if handled:
                return

        # add to conversation history
        self.messages.append({"role": "user", "content": stripped})

        # build additional context (thesis + RAG)
        context_messages: List[Dict[str, Any]] = []

        # 1) thesis section: highest priority
        if self.current_thesis_section_text:
            context_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Here is the current section of my thesis that should guide your answer. "
                        "Please edit only this section unless I say otherwise:\n\n"
                        + self.current_thesis_section_text
                    ),
                }
            )

        # 2) RAG context
        if self.citations_enabled:
            results = self.dm.search(stripped, top_k=DEFAULT_TOP_K)
            self.update_snippets_from_results(results)
            rag_context = self.build_rag_context(results)
            if rag_context:
                context_messages.append(
                    {
                        "role": "user",
                        "content": rag_context,
                    }
                )

        # messages for model = history + context + last user query (again)
        history = self.get_limited_history()
        messages_for_model = history + context_messages + [{"role": "user", "content": stripped}]

        # call model in background
        self.show_typing_indicator(True)
        self.set_send_button_pressed(True)
        self.start_model_thread(messages_for_model)

    # ---------- background model calls ----------

    def start_model_thread(self, messages: List[Dict[str, Any]]):
        def worker():
            # local copy so we don't mutate self.messages from this thread
            model_messages = list(messages)
            final_content = ""

            while True:
                try:
                    resp = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=model_messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                except Exception as e:
                    self.model_error.emit(str(e))
                    return

                choice = resp.choices[0]
                msg = choice.message

                tool_calls = getattr(msg, "tool_calls", None) or []

                # No tool calls -> we got a final answer
                if not tool_calls:
                    final_content = msg.content or ""
                    break

                # Record the assistant's tool request in the local transcript
                model_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                # Execute each tool call
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    # Dispatch
                    if name == "read_thesis_metadata":
                        result = self.tool_read_thesis_metadata()
                    elif name == "read_thesis_section":
                        idx = int(args.get("index", -1))
                        result = self.tool_read_thesis_section(idx)
                    elif name == "write_thesis_section":
                        idx = int(args.get("index", -1))
                        new_text = args.get("new_text", "")
                        result = self.tool_write_thesis_section(idx, new_text)
                    else:
                        result = {"error": f"Unknown tool {name}"}

                    # Append tool result message
                    model_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )

            # At this point we have a final answer
            self.model_reply_ready.emit(final_content)

        t = threading.Thread(target=worker, daemon=True)
        t.start()


    def on_model_reply_ready(self, content: str):
        self.show_typing_indicator(False)
        self.set_send_button_pressed(False)
        if content.strip():
            self.messages.append({"role": "assistant", "content": content})
            self.append_assistant_message(escape_html(content))
            for eq_full in self.extract_equations(content):
                if eq_full not in self.displayed_equations:
                    self.displayed_equations.append(eq_full)
            self.update_right_panel()

    def on_model_error(self, err: str):
        self.show_typing_indicator(False)
        self.set_send_button_pressed(False)
        msg = f"Error contacting model: {err}"
        self.append_assistant_message(escape_html(msg))
        print(msg)

    # ---------- command-mode ----------

    def handle_command(self, cmd: str) -> bool:
        """
        Handle meta commands like :help, :pipeline, :thesis-*, :tex-*.
        Returns True if handled (no model call).
        """
        parts = cmd.split(maxsplit=1)
        name = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if name in (":help", ":h"):
            help_text = (
                "<b>Commands:</b><br>"
                "<code>:help</code> – show this help<br>"
                "<code>:pipeline</code> – run update_metadata → process_pdfs → chunk_text_files → process_embeddings<br>"
                "<br>"
                "<b>Thesis-specific:</b><br>"
                "<code>:thesis-open [path]</code> – load Thesis.tex (defaults to DATA_SERVER/thesis/Thesis.tex)<br>"
                "<code>:thesis-section &lt;pattern&gt;</code> – focus on section/subsection whose title contains pattern<br>"
                "<code>:thesis-clear</code> – clear current section context<br>"
                "<code>:thesis-apply</code> – replace current section in file with last assistant message<br>"
                "<code>:thesis-compile</code> – run pdflatex and open PDF<br>"
                "<code>:thesis-pdf</code> – open compiled thesis PDF<br>"
                "<br>"
                "<b>Generic TeX commands:</b><br>"
                "<code>:tex-open [path]</code> – load any .tex file<br>"
                "<code>:tex-reload</code> – reload from disk<br>"
                "<code>:tex-path</code> – show current TeX file path<br>"
                "<code>:tex-show [N]</code> – show first N lines (default 40)<br>"
                "<code>:tex-save</code> – save full TeX buffer to disk<br>"
                "<code>:tex-replace</code> – replace entire TeX file with last assistant message<br>"
            )
            self.append_assistant_message(help_text)
            return True

        if name == ":pipeline":
            self.append_assistant_message(
                "Running pipeline: update_metadata → process_pdfs → chunk_text_files → process_embeddings..."
            )
            try:
                run_pipeline(self.dm)
                self.append_assistant_message("Pipeline completed.")
            except Exception as e:
                self.append_assistant_message(
                    escape_html(f"Pipeline error: {e}")
                )
            return True

        # ----- Thesis commands -----

        if name == ":thesis-open":
            self.load_thesis_from_default(arg)
            return True

        if name == ":thesis-section":
            self.select_thesis_section(arg)
            return True

        if name == ":thesis-clear":
            self.current_thesis_section = None
            self.current_thesis_section_text = ""
            self.append_assistant_message("Cleared current thesis section context.")
            return True

        if name == ":thesis-apply":
            self.apply_last_answer_to_current_section()
            return True

        if name == ":thesis-compile":
            self.compile_thesis()
            return True

        if name == ":thesis-pdf":
            self.open_thesis_pdf()
            return True

        # ----- generic TeX commands -----

        if name == ":tex-open":
            if arg:
                path = os.path.expanduser(arg)
            else:
                dlg = QFileDialog(self, "Open LaTeX file", os.getcwd(), "TeX Files (*.tex)")
                dlg.setFileMode(QFileDialog.ExistingFile)
                if dlg.exec_() != QFileDialog.Accepted:
                    return True
                path = dlg.selectedFiles()[0]
            self.load_tex_file(path, treat_as_thesis=False)
            return True

        if name == ":tex-reload":
            self.reload_tex_file()
            return True

        if name == ":tex-path":
            if not self.current_tex_path:
                self.append_assistant_message("No TeX file loaded.")
            else:
                self.append_assistant_message(
                    escape_html(f"Current TeX file: {self.current_tex_path}")
                )
            return True

        if name == ":tex-show":
            if not self.current_tex_content:
                self.append_assistant_message("No TeX file loaded.")
                return True
            try:
                n = int(arg) if arg else 40
            except Exception:
                n = 40
            lines = self.current_tex_content.splitlines()
            snippet = "\n".join(lines[:n])
            html = (
                f"First {min(n, len(lines))} lines of TeX:<br>"
                f"<pre><code>{escape_html(snippet)}</code></pre>"
            )
            self.append_assistant_message(html)
            return True

        if name == ":tex-save":
            if not self.current_tex_path:
                self.append_assistant_message("No TeX file loaded.")
                return True
            try:
                with open(self.current_tex_path, "w", encoding="utf-8") as f:
                    f.write(self.current_tex_content)
                self.append_assistant_message(
                    escape_html(f"Saved TeX file: {self.current_tex_path}")
                )
            except Exception as e:
                self.append_assistant_message(
                    escape_html(f"Error saving TeX file: {e}")
                )
            return True

        if name == ":tex-replace":
            if not self.current_tex_path:
                self.append_assistant_message(
                    "No TeX file loaded. Use <code>:tex-open</code> first."
                )
                return True

            last_assistant = None
            for m in reversed(self.messages):
                if m.get("role") == "assistant" and m.get("content"):
                    last_assistant = m["content"]
                    break

            if not last_assistant:
                self.append_assistant_message(
                    "No assistant message found to use as TeX content."
                )
                return True

            try:
                self.current_tex_content = last_assistant
                with open(self.current_tex_path, "w", encoding="utf-8") as f:
                    f.write(self.current_tex_content)
                self.append_assistant_message(
                    escape_html(
                        f"Replaced contents of {self.current_tex_path} with last assistant message "
                        f"({len(self.current_tex_content.splitlines())} lines)."
                    )
                )
                # reparse sections if this is the thesis
                if self.current_tex_path and os.path.dirname(self.current_tex_path) == os.path.abspath(THESIS_DIR):
                    self.parse_thesis_sections()
            except Exception as e:
                self.append_assistant_message(
                    escape_html(f"Error writing TeX file: {e}")
                )
            return True

        # unknown command
        self.append_assistant_message(
            escape_html(f"Unknown command: {name}. Try :help.")
        )
        return True

    # ---------- thesis / TeX helpers ----------

    def load_thesis_from_default(self, arg_path: str = ""):
        """Load Thesis.tex from THESIS_DIR or a provided path."""
        if arg_path:
            path = os.path.expanduser(arg_path)
        else:
            default_path = os.path.join(THESIS_DIR, "Thesis.tex")
            if os.path.exists(default_path):
                path = default_path
            else:
                dlg = QFileDialog(
                    self,
                    "Open Thesis.tex",
                    THESIS_DIR,
                    "TeX Files (*.tex)",
                )
                dlg.setFileMode(QFileDialog.ExistingFile)
                if dlg.exec_() != QFileDialog.Accepted:
                    return
                path = dlg.selectedFiles()[0]
        self.load_tex_file(path, treat_as_thesis=True)

    def load_tex_file(self, path: str, treat_as_thesis: bool):
        if not os.path.exists(path):
            self.append_assistant_message(escape_html(f"No such file: {path}"))
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                self.current_tex_content = f.read()
            self.current_tex_path = os.path.abspath(path)
            n_lines = len(self.current_tex_content.splitlines())
            role = "thesis" if treat_as_thesis else "TeX"
            self.append_assistant_message(
                escape_html(f"Loaded {role} file: {self.current_tex_path} ({n_lines} lines).")
            )

            if treat_as_thesis:
                self.parse_thesis_sections()
                self.current_thesis_section = None
                self.current_thesis_section_text = ""
                self.messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"The user has loaded their thesis from '{self.current_tex_path}'. "
                            "When they ask for edits, assume this is the main document. "
                            "Use the current section context when provided."
                        ),
                    }
                )
        except Exception as e:
            self.append_assistant_message(
                escape_html(f"Error loading TeX file: {e}")
            )

    def reload_tex_file(self):
        if not self.current_tex_path:
            self.append_assistant_message("No TeX file loaded.")
            return
        path = self.current_tex_path
        if not os.path.exists(path):
            self.append_assistant_message(
                escape_html(f"File no longer exists: {path}")
            )
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.current_tex_content = f.read()
            n_lines = len(self.current_tex_content.splitlines())
            self.append_assistant_message(
                escape_html(f"Reloaded TeX file: {self.current_tex_path} ({n_lines} lines).")
            )
            if os.path.dirname(path) == os.path.abspath(THESIS_DIR):
                self.parse_thesis_sections()
        except Exception as e:
            self.append_assistant_message(
                escape_html(f"Error reloading TeX file: {e}")
            )

    def parse_thesis_sections(self):
        """Parse \section and \subsection boundaries for the current thesis."""
        self.thesis_sections = []
        text = self.current_tex_content
        if not text:
            return

        pattern = re.compile(r"\\(sub)*section\{([^}]*)\}")
        matches = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            level = "subsection" if m.group(1) == "sub" else "section"
            title = m.group(2)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            self.thesis_sections.append(
                {"level": level, "title": title, "start": start, "end": end}
            )

    def select_thesis_section(self, pattern: str):
        if not self.current_tex_content:
            self.append_assistant_message(
                "No thesis file loaded. Use <code>:thesis-open</code> first."
            )
            return
        if not pattern:
            self.append_assistant_message(
                "Usage: <code>:thesis-section &lt;pattern&gt;</code>."
            )
            return

        if not self.thesis_sections:
            self.parse_thesis_sections()

        patt = pattern.lower()
        chosen = None
        for sec in self.thesis_sections:
            if patt in sec["title"].lower():
                chosen = sec
                break

        if not chosen:
            self.append_assistant_message(
                escape_html(f"No section or subsection title contains '{pattern}'.")
            )
            return

        self.current_thesis_section = chosen
        start, end = chosen["start"], chosen["end"]
        self.current_thesis_section_text = self.current_tex_content[start:end]

        preview = "\n".join(self.current_thesis_section_text.splitlines()[:20])
        msg = (
            f"Selected {chosen['level']} '{chosen['title']}' as current thesis section context.\n"
            "I will prioritize this section in my answers.\n\n"
            "Preview (first ~20 lines):\n\n" + preview
        )
        self.append_assistant_message(escape_html(msg))

    def apply_last_answer_to_current_section(self):
        if not self.current_tex_path:
            self.append_assistant_message(
                "No thesis file loaded. Use <code>:thesis-open</code> first."
            )
            return
        if not self.current_thesis_section:
            self.append_assistant_message(
                "No current thesis section selected. Use <code>:thesis-section</code> first."
            )
            return

        last_assistant = None
        for m in reversed(self.messages):
            if m.get("role") == "assistant" and m.get("content"):
                last_assistant = m["content"]
                break

        if not last_assistant:
            self.append_assistant_message(
                "No assistant message found to apply to the thesis."
            )
            return

        sec = self.current_thesis_section
        start, end = sec["start"], sec["end"]

        try:
            # splice new content into TeX buffer
            self.current_tex_content = (
                self.current_tex_content[:start]
                + last_assistant
                + self.current_tex_content[end:]
            )
            with open(self.current_tex_path, "w", encoding="utf-8") as f:
                f.write(self.current_tex_content)

            # reparse sections and reselect by title
            old_title = sec["title"]
            self.parse_thesis_sections()
            self.current_thesis_section = None
            self.current_thesis_section_text = ""

            for new_sec in self.thesis_sections:
                if new_sec["title"] == old_title and new_sec["level"] == sec["level"]:
                    self.current_thesis_section = new_sec
                    s2, e2 = new_sec["start"], new_sec["end"]
                    self.current_thesis_section_text = self.current_tex_content[s2:e2]
                    break

            self.append_assistant_message(
                escape_html(
                    f"Replaced {sec['level']} '{old_title}' in {self.current_tex_path} "
                    f"with last assistant message."
                )
            )
        except Exception as e:
            self.append_assistant_message(
                escape_html(f"Error applying section update: {e}")
            )

    def compile_thesis(self):
        if not self.current_tex_path:
            self.append_assistant_message(
                "No TeX file loaded. Use <code>:thesis-open</code> or the 'Load Thesis' button."
            )
            return

        tex_path = self.current_tex_path
        dirpath = os.path.dirname(tex_path)
        filename = os.path.basename(tex_path)

        self.append_assistant_message(
            escape_html(f"Compiling '{filename}' with pdflatex...")
        )

        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", filename],
                cwd=dirpath,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0:
                log_lines = result.stdout.splitlines()
                tail = "\n".join(log_lines[-30:])
                self.append_assistant_message(
                    escape_html(
                        "pdflatex failed. Last ~30 lines of log:\n\n" + tail
                    )
                )
            else:
                self.append_assistant_message(
                    escape_html("Compilation finished. Opening PDF...")
                )
                self.open_thesis_pdf()
        except FileNotFoundError:
            self.append_assistant_message(
                "pdflatex not found on PATH. Install TeXLive or ensure pdflatex is in your PATH."
            )
        except Exception as e:
            self.append_assistant_message(
                escape_html(f"Error during compilation: {e}")
            )

    def open_thesis_pdf(self):
        if not self.current_tex_path:
            self.append_assistant_message(
                "No TeX file loaded. Use <code>:thesis-open</code> or the 'Load Thesis' button."
            )
            return
        pdf_path = os.path.splitext(self.current_tex_path)[0] + ".pdf"
        if not os.path.exists(pdf_path):
            self.append_assistant_message(
                escape_html(f"PDF not found: {pdf_path}. Try compiling first.")
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))

    # ---------- tool execution for thesis operations ----------

    def tool_read_thesis_metadata(self) -> Dict[str, Any]:
        """
        Return a simple description of the thesis sections/subsections:
          { "path": "...", "sections": [ { "index": i, "level": "section"/"subsection", "title": "..." }, ... ] }
        """
        if not self.current_tex_path:
            return {
                "error": "No thesis file loaded. Use :thesis-open or the 'Load Thesis' button first."
            }

        # Make sure sections are parsed
        if not self.thesis_sections:
            self.parse_thesis_sections()

        sections = []
        for i, sec in enumerate(self.thesis_sections):
            sections.append(
                {
                    "index": i,
                    "level": sec["level"],
                    "title": sec["title"],
                }
            )

        return {
            "path": self.current_tex_path,
            "sections": sections,
        }

    def tool_read_thesis_section(self, index: int) -> Dict[str, Any]:
        """
        Return the LaTeX source for a specific section/subsection.
        Also sets current_thesis_section/current_thesis_section_text for the GUI.
        """
        if not self.current_tex_path:
            return {
                "error": "No thesis file loaded. Use :thesis-open or the 'Load Thesis' button first."
            }

        if not self.thesis_sections:
            self.parse_thesis_sections()

        if index < 0 or index >= len(self.thesis_sections):
            return {
                "error": f"Invalid section index {index}. There are only {len(self.thesis_sections)} sections.",
            }

        sec = self.thesis_sections[index]
        start, end = sec["start"], sec["end"]
        text = self.current_tex_content[start:end]

        # update GUI's notion of current section
        self.current_thesis_section = sec
        self.current_thesis_section_text = text

        return {
            "index": index,
            "level": sec["level"],
            "title": sec["title"],
            "text": text,
        }

    def tool_write_thesis_section(self, index: int, new_text: str) -> Dict[str, Any]:
        """
        Replace the LaTeX source of a specific section/subsection with new_text and save the file.
        """
        if not self.current_tex_path:
            return {
                "error": "No thesis file loaded. Use :thesis-open or the 'Load Thesis' button first."
            }

        if not self.thesis_sections:
            self.parse_thesis_sections()

        if index < 0 or index >= len(self.thesis_sections):
            return {
                "error": f"Invalid section index {index}. There are only {len(self.thesis_sections)} sections.",
            }

        sec = self.thesis_sections[index]
        start, end = sec["start"], sec["end"]

        try:
            # splice new content
            self.current_tex_content = (
                self.current_tex_content[:start] + new_text + self.current_tex_content[end:]
            )
            # write to disk
            with open(self.current_tex_path, "w", encoding="utf-8") as f:
                f.write(self.current_tex_content)

            # reparse all sections so indices/positions stay correct
            old_title = sec["title"]
            old_level = sec["level"]
            self.parse_thesis_sections()

            # try to reselect the same section by title+level
            new_sec = None
            for s in self.thesis_sections:
                if s["title"] == old_title and s["level"] == old_level:
                    new_sec = s
                    break

            if new_sec is not None:
                self.current_thesis_section = new_sec
                self.current_thesis_section_text = self.current_tex_content[
                    new_sec["start"] : new_sec["end"]
                ]
            else:
                self.current_thesis_section = None
                self.current_thesis_section_text = ""

            return {
                "status": "ok",
                "message": f"Replaced {sec['level']} '{sec['title']}' in {self.current_tex_path}.",
            }

        except Exception as e:
            return {
                "error": f"Error writing section: {e}",
            }


    # ---------- RAG helpers ----------

    def update_snippets_from_results(self, results: List[dict]):
        for r in results:
            snippet = self.get_snippet(r)
            pdf_name = r["pdf_filename"]
            citation_key = r.get("references", {}).get("citation_key", "unknown_key")

            key = (pdf_name, snippet)
            if key in self.snippet_cache:
                continue

            snippet_id = len(self.indexed_snippets) + 1
            self.indexed_snippets.append(
                {
                    "id": snippet_id,
                    "citation_key": citation_key,
                    "pdf_name": pdf_name,
                    "snippet": snippet,
                }
            )
            self.snippet_cache[key] = snippet_id

        self.update_right_panel()

    def build_rag_context(self, results: List[dict]) -> str:
        if not results:
            return ""
        lines = [
            "Here are some relevant references from my local corpus. "
            "Use them for factual grounding and citations. "
            "When you cite them, use the given citation_key with LaTeX \\cite{citation_key}.\n"
        ]
        for i, r in enumerate(results, start=1):
            refs = r.get("references") or {}
            citation_key = refs.get("citation_key", "unknown_key")
            pdf_name = r["pdf_filename"]
            snippet = self.get_snippet(r)
            lines.append(
                f"[{i}] {citation_key} ({pdf_name})\n{snippet}\n"
            )
        return "\n".join(lines)

    def get_snippet(self, result: dict) -> str:
        chunk_file = result["chunk_filename"]
        chunk_path = os.path.join(self.dm.chunk_files_directory, chunk_file)
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return "Could not read chunk."

    def extract_equations(self, text: str) -> List[str]:
        pattern = r"\\begin{equation}.*?\\end{equation}"
        return [m.group(0) for m in re.finditer(pattern, text, flags=re.DOTALL)]

    def update_right_panel(self):
        # clear
        for i in reversed(range(self.right_inner_layout.count())):
            w = self.right_inner_layout.itemAt(i).widget()
            if w is not None:
                w.deleteLater()

        # thesis section indicator
        if self.current_thesis_section:
            lbl = QLabel(
                f"Thesis context: {self.current_thesis_section['level']} "
                f"'{self.current_thesis_section['title']}'"
            )
            lbl.setFont(QFont("Arial", 10, QFont.Bold))
            self.right_inner_layout.addWidget(lbl)

        # snippets
        for s in self.indexed_snippets:
            frame = QFrame()
            layout = QHBoxLayout(frame)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(5)

            id_label = QLabel(f"[{s['id']}]")
            id_label.setFont(QFont("Arial", 10, QFont.Bold))
            layout.addWidget(id_label)

            key_label = QLabel(s["citation_key"])
            key_label.setFont(QFont("Arial", 10))
            layout.addWidget(key_label)

            cite_button = QPushButton("Copy \\cite")
            cite_button.setFont(QFont("Arial", 10))
            citation_text = f"\\cite{{{s['citation_key']}}}"
            cite_button.clicked.connect(
                lambda _ch, c=citation_text: self.copy_to_clipboard(c)
            )
            layout.addWidget(cite_button)

            snippet_button = QPushButton("Snippet")
            snippet_button.setFont(QFont("Arial", 10))
            snippet_button.clicked.connect(
                lambda _ch, sn=s: self.show_snippet_dialog(sn["snippet"], sn["pdf_name"])
            )
            layout.addWidget(snippet_button)

            pdf_button = QPushButton("Open PDF")
            pdf_button.setFont(QFont("Arial", 10))
            pdf_button.clicked.connect(
                lambda _ch, pdfn=s["pdf_name"]: self.open_pdf(pdfn)
            )
            layout.addWidget(pdf_button)

            self.right_inner_layout.addWidget(frame)

        # equations
        for eq_full in self.displayed_equations:
            eq_frame = QFrame()
            eq_layout = QHBoxLayout(eq_frame)
            eq_layout.setContentsMargins(0, 0, 0, 0)
            eq_layout.setSpacing(5)

            eq_label = QLabel("equation")
            eq_label.setFont(QFont("Arial", 10, QFont.Bold))
            eq_layout.addWidget(eq_label)

            copy_button = QPushButton("Copy")
            copy_button.setFont(QFont("Arial", 10))
            copy_button.clicked.connect(
                lambda _ch, eq=eq_full: self.copy_to_clipboard(eq)
            )
            eq_layout.addWidget(copy_button)

            self.right_inner_layout.addWidget(eq_frame)

        self.right_inner_layout.addStretch(1)

    def copy_to_clipboard(self, text: str):
        pyperclip.copy(text)
        print("Copied to clipboard:", text)

    def show_snippet_dialog(self, snippet: str, pdf_name: str):
        dlg = SnippetDialog(snippet, pdf_name, parent=self)
        dlg.exec_()

    def open_pdf(self, pdf_filename: str):
        pdf_path = os.path.join(self.dm.pdf_files_directory, pdf_filename)
        if os.path.exists(pdf_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))
        else:
            print("PDF file not found:", pdf_path)

    # ---------- sessions save/load ----------

    def save_session(self):
        session_data = {
            "messages": self.messages,
            "equations": self.displayed_equations,
            "snippets": self.indexed_snippets,
        }
        timestamp = int(time.time())
        save_path = os.path.join(APP_SESSIONS_PATH, f"session_{timestamp}.json")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)
            print(f"Session saved to {save_path}")
        except Exception as e:
            print("Error saving session:", e)

    def load_session(self):
        dlg = QFileDialog(self, "Load Session", APP_SESSIONS_PATH, "JSON Files (*.json)")
        dlg.setFileMode(QFileDialog.ExistingFile)
        if dlg.exec_() != QFileDialog.Accepted:
            return
        file_path = dlg.selectedFiles()[0]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)
        except Exception as e:
            print("Error loading session:", e)
            return

        self.messages = session_data.get("messages", [])
        self.displayed_equations = session_data.get("equations", [])
        self.indexed_snippets = session_data.get("snippets", [])

        self.snippet_cache = {}
        for s in self.indexed_snippets:
            key = (s["pdf_name"], s["snippet"])
            self.snippet_cache[key] = s["id"]

        self.html_messages = []
        for m in self.messages:
            role = m.get("role", "assistant")
            cont = m.get("content", "")
            if not cont or not cont.strip():
                continue
            if role == "user":
                self.html_messages.append(
                    f'<div class="message-container"><div class="message-user"><b>You:</b><br>{escape_html(cont)}</div></div>'
                )
            elif role == "assistant":
                self.html_messages.append(
                    f'<div class="message-container"><div class="message-assistant"><b>GPT:</b><br>{escape_html(cont)}</div></div>'
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
    window.show()
    sys.exit(app.exec_())
