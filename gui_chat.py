import sys
import os
import json
import time
import re
import threading
import subprocess
import shutil
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QFont, QKeyEvent, QKeySequence, QDesktopServices
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QFrame,
    QFileDialog,
    QShortcut,
    QCheckBox,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView

from openai import OpenAI

from data_manager import DataManager, run_pipeline


# ---------- config & helpers ----------

load_dotenv()


def _env(primary: str, *alts: str, default: Optional[str] = None) -> Optional[str]:
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

# Optional override for bib path
THESIS_BIB_PATH = os.getenv("THESIS_BIB_PATH", "")

CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", "96000"))
CONTEXT_TRIGGER_TOKENS = int(CONTEXT_TOKEN_LIMIT * 0.8)
COMPRESS_KEEP_LAST = int(os.getenv("COMPRESS_KEEP_LAST", "12"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "20"))

client = OpenAI()


def escape_html(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\n", "<br>")
    return text


def estimate_tokens_for_messages(messages: List[Dict[str, Any]]) -> int:
    total_chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total_chars += len(content)
    return total_chars // 4


def compress_history_if_needed(
    messages: List[Dict[str, Any]],
    client: OpenAI,
    keep_last: int = COMPRESS_KEEP_LAST,
) -> List[Dict[str, Any]]:
    est_tokens = estimate_tokens_for_messages(messages)
    if est_tokens <= CONTEXT_TRIGGER_TOKENS:
        return list(messages)

    if len(messages) <= keep_last + 2:
        return list(messages)

    summarizable = messages[:-keep_last]
    tail = messages[-keep_last:]

    to_summarize = [
        m for m in summarizable
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    if not to_summarize:
        return list(messages)

    summary_prompt = [
        {
            "role": "system",
            "content": (
                "You are summarizing an earlier technical conversation between a user and an assistant.\n"
                "Produce a concise summary (300–500 words) that preserves:\n"
                "- key definitions, equations, and LaTeX snippets\n"
                "- important decisions or conclusions\n"
                "- open questions or TODOs.\n"
                "Do NOT include any meta-comments about tools or the system; just summarize the content."
            ),
        }
    ] + to_summarize

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=summary_prompt,
        )
        summary_text = resp.choices[0].message.content or ""
    except Exception:
        return list(messages)

    new_messages: List[Dict[str, Any]] = [
        m for m in messages if m.get("role") == "system"
    ]
    new_messages.append(
        {
            "role": "system",
            "content": "Summary of earlier conversation (for context):\n" + summary_text,
        }
    )
    new_messages.extend(tail)
    return new_messages


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_thesis_metadata",
            "description": "List sections/subsections of the loaded thesis file.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_thesis_section",
            "description": "Return LaTeX source of a specific section/subsection by index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Index from read_thesis_metadata (0-based).",
                    }
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_thesis_section",
            "description": "Overwrite a section/subsection with new LaTeX after user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "new_text": {"type": "string"},
                },
                "required": ["index", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_corpus",
            "description": (
                "Search the local PDF corpus for relevant references. "
                "Returns suggested citation keys and snippets; you must cross-check these "
                "against the BibTeX index before using them in \\cite{...}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": RAG_TOP_K,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_bib_index",
            "description": (
                "Read the thesis BibTeX file (references.bib or THESIS_BIB_PATH) "
                "and return a list of entries (key + title). "
                "Only use keys from this list in LaTeX \\cite{...}."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_bib_entry",
            "description": (
                "Return the full BibTeX entry for a given key from the thesis bibliography."
            ),
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
]


class InputTextEdit(QTextEdit):
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
    model_reply_ready = pyqtSignal(str)
    model_error = pyqtSignal(str)

    def __init__(self, dm: DataManager):
        super().__init__()
        self.dm = dm

        self.messages: List[Dict[str, Any]] = []
        self.html_messages: List[str] = []
        self.typing_indicator_visible = False

        self.citations_enabled = True

        self.current_tex_path: Optional[str] = None
        self.current_tex_content: str = ""
        self.thesis_sections: List[Dict[str, Any]] = []

        self._bib_cache: Dict[str, str] = {}
        self._bib_cache_path: Optional[str] = None

        self.send_button_normal_style = (
            "background-color: #4CAF50; color: white; padding: 6px 12px;"
        )
        self.send_button_pressed_style = (
            "background-color: #4CAF50; color: white; padding: 6px 12px; margin-left: 10px;"
        )

        self.model_reply_ready.connect(self.on_model_reply_ready)
        self.model_error.connect(self.on_model_error)

        self.init_ui()
        self.new_session()

    # ---------- UI ----------

    def init_ui(self):
        self.setWindowTitle("GPT Reference Assistant")

        self.setStyleSheet(
            """
        QWidget { background-color: #202020; color: #ffffff; }
        QTextEdit { background-color: #2a2a2a; color: #ffffff; }
        QPushButton {
            background-color: #4CAF50;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QPushButton:hover { background-color: #45a049; }
        """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        menu_layout = QHBoxLayout()

        new_button = QPushButton("New")
        new_button.clicked.connect(self.new_session)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_session)

        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_session)

        self.citations_checkbox = QCheckBox("Use corpus for citations")
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

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addLayout(content_layout)

        title_label = QLabel("GPT Reference Assistant")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignHCenter)
        title_label.setStyleSheet("color: #4CAF50;")
        content_layout.addWidget(title_label)

        self.web_view = QWebEngineView(self)
        content_layout.addWidget(self.web_view)

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
        content_layout.addWidget(input_frame)

        content_layout.setStretch(0, 0)  # title
        content_layout.setStretch(1, 1)  # webview
        content_layout.setStretch(2, 0)  # input

        self.shortcut_new = QShortcut(QKeySequence("Ctrl+N"), self)
        self.shortcut_new.activated.connect(self.new_session)
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_session)
        self.shortcut_load = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_load.activated.connect(self.load_session)

        self.resize(1200, 900)
        self.update_html()
        self.showMaximized()

    # ---------- chat helpers ----------

    def show_typing_indicator(self, show: bool = True):
        self.typing_indicator_visible = show
        self.update_html()

    def set_send_button_pressed(self, pressed: bool = True):
        self.send_button.setEnabled(not pressed)
        self.send_button.setStyleSheet(
            self.send_button_pressed_style if pressed else self.send_button_normal_style
        )

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
.message-container { margin-bottom: 10px; }
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
window.MathJax = { tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]} };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"
        id="MathJax" async></script>
</head>
<body>
"""
        html_body = "".join(self.html_messages)
        if self.typing_indicator_visible:
            html_body += '<div class="typing">GPT is typing...</div>'
        html_footer = "</body></html>"
        self.web_view.setHtml(html_head + html_body + html_footer)
        self.web_view.page().runJavaScript(
            "window.scrollTo(0, document.body.scrollHeight);"
        )

    # ---------- session ----------

    def new_session(self):
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant helping with a LaTeX-based physics thesis.\n"
                    "- The user has a local PDF corpus. When you need references or citations, "
                    "call `search_local_corpus` to get suggested citation keys and snippets.\n"
                    "- The thesis BibTeX file is the source of truth for citation keys. "
                    "Before inserting or modifying any \\cite{key}, call `read_bib_index` to see which keys exist. "
                    "Optionally call `read_bib_entry` for details. Never invent new keys.\n"
                    "- The main thesis is in a LaTeX file on disk. Use tools:\n"
                    "  * read_thesis_metadata → list sections/subsections\n"
                    "  * read_thesis_section → read a specific section\n"
                    "  * write_thesis_section → overwrite a section after explicit user approval\n"
                    "- Do not narrate tool usage. Just provide the resulting LaTeX and a brief description of changes."
                ),
            }
        ]
        self.html_messages = []
        self.typing_indicator_visible = False
        self.update_html()
        self.set_send_button_pressed(False)
        print("New session started.")

    # ---------- sending / commands ----------

    def send_message(self):
        raw_text = self.input_text.toPlainText()
        stripped = raw_text.strip()
        if not stripped:
            return

        self.append_user_message(escape_html(raw_text))
        self.input_text.clear()

        if stripped.startswith(":"):
            handled = self.handle_command(stripped)
            if handled:
                return

        history = list(self.messages)
        messages_for_model = history + [{"role": "user", "content": stripped}]
        self.messages.append({"role": "user", "content": stripped})

        self.show_typing_indicator(True)
        self.set_send_button_pressed(True)
        self.start_model_thread(messages_for_model)

    # ---------- background model calls ----------

    def start_model_thread(self, messages: List[Dict[str, Any]]):
        def worker():
            model_messages = compress_history_if_needed(messages, client)
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

                if not tool_calls:
                    final_content = msg.content or ""
                    break

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

                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    if name == "read_thesis_metadata":
                        result = self.tool_read_thesis_metadata()
                    elif name == "read_thesis_section":
                        idx = int(args.get("index", -1))
                        result = self.tool_read_thesis_section(idx)
                    elif name == "write_thesis_section":
                        idx = int(args.get("index", -1))
                        new_text = args.get("new_text", "")
                        result = self.tool_write_thesis_section(idx, new_text)
                    elif name == "search_local_corpus":
                        query = args.get("query", "")
                        top_k = int(args.get("top_k", RAG_TOP_K))
                        result = self.tool_search_local_corpus(query, top_k)
                    elif name == "read_bib_index":
                        result = self.tool_read_bib_index()
                    elif name == "read_bib_entry":
                        key = args.get("key", "")
                        result = self.tool_read_bib_entry(key)
                    else:
                        result = {"error": f"Unknown tool {name}"}

                    model_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )

            self.model_reply_ready.emit(final_content)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def on_model_reply_ready(self, content: str):
        self.show_typing_indicator(False)
        self.set_send_button_pressed(False)
        if content.strip():
            self.messages.append({"role": "assistant", "content": content})
            self.append_assistant_message(escape_html(content))

    def on_model_error(self, err: str):
        self.show_typing_indicator(False)
        self.set_send_button_pressed(False)
        msg = f"Error contacting model: {err}"
        self.append_assistant_message(escape_html(msg))
        print(msg)

    # ---------- commands ----------

    def handle_command(self, cmd: str) -> bool:
        parts = cmd.split(maxsplit=1)
        name = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if name in (":help", ":h"):
            help_text = (
                "<b>Commands:</b><br>"
                "<code>:help</code> – show this help<br>"
                "<code>:pipeline</code> – run update_metadata → process_pdfs → chunk_text_files → process_embeddings<br>"
                "<br><b>Thesis:</b><br>"
                "<code>:thesis-open [path]</code> – load Thesis.tex (default DATA_SERVER/thesis/Thesis.tex)<br>"
                "<code>:thesis-compile</code> – compile via latexmk/pdflatex+bibtex<br>"
                "<code>:thesis-pdf</code> – open compiled PDF<br>"
                "<br><b>TeX:</b><br>"
                "<code>:tex-open [path]</code>, <code>:tex-reload</code>, <code>:tex-path</code>, "
                "<code>:tex-show [N]</code>, <code>:tex-save</code>, <code>:tex-replace</code><br>"
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
                self.append_assistant_message(escape_html(f"Pipeline error: {e}"))
            return True

        if name == ":thesis-open":
            self.load_thesis_from_default(arg)
            return True

        if name == ":thesis-compile":
            self.compile_thesis()
            return True

        if name == ":thesis-pdf":
            self.open_thesis_pdf()
            return True

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
                if os.path.dirname(self.current_tex_path) == os.path.abspath(THESIS_DIR):
                    self.parse_thesis_sections()
            except Exception as e:
                self.append_assistant_message(
                    escape_html(f"Error writing TeX file: {e}")
                )
            return True

        self.append_assistant_message(
            escape_html(f"Unknown command: {name}. Try :help.")
        )
        return True

    # ---------- thesis / TeX helpers ----------

    def load_thesis_from_default(self, arg_path: str = ""):
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
                self.messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"The user has loaded their thesis from '{self.current_tex_path}'. "
                            "Use the thesis tools when editing sections."
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
        """
        Parse \section / \subsection only in the 'body' before the bibliography and \end{document},
        so edits never overwrite the references or \end{document}.
        """
        self.thesis_sections = []
        text = self.current_tex_content
        if not text:
            return

        # Find \end{document}
        enddoc_match = re.search(r"\\end\{document\}", text)
        if enddoc_match:
            enddoc_idx = enddoc_match.start()
            body_pre_end = text[:enddoc_idx]
        else:
            enddoc_idx = len(text)
            body_pre_end = text

        # Find bibliography start in body_pre_end
        refs_match_1 = re.search(r"\\bibliography\{", body_pre_end)
        refs_match_2 = re.search(r"\\begin\{thebibliography\}", body_pre_end)
        candidates = [m.start() for m in (refs_match_1, refs_match_2) if m]
        refs_start_idx = min(candidates) if candidates else len(body_pre_end)

        # Body in which we allow section edits
        body_main = body_pre_end[:refs_start_idx]

        pattern = re.compile(r"\\(sub)*section\{([^}]*)\}")
        matches = list(pattern.finditer(body_main))

        for i, m in enumerate(matches):
            level = "subsection" if m.group(1) == "sub" else "section"
            title = m.group(2)
            start = m.start()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(body_main)
            # indices in body_main coincide with indices in full text (since body_main is prefix)
            self.thesis_sections.append(
                {"level": level, "title": title, "start": start, "end": end}
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
        basename = os.path.splitext(filename)[0]

        self.append_assistant_message(
            escape_html(f"Compiling '{filename}' from directory:\n{dirpath}")
        )

        try:
            if shutil.which("latexmk"):
                cmd = [
                    "latexmk",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    filename,
                ]
                result = subprocess.run(
                    cmd,
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
                            "latexmk failed. Last ~30 lines of log:\n\n" + tail
                        )
                    )
                    return
            else:
                logs = []

                def run_cmd(cmd_list):
                    r = subprocess.run(
                        cmd_list,
                        cwd=dirpath,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    logs.append((cmd_list, r.returncode, r.stdout))
                    return r

                r1 = run_cmd(["pdflatex", "-interaction=nonstopmode", filename])
                r2 = run_cmd(["bibtex", basename])
                r3 = run_cmd(["pdflatex", "-interaction=nonstopmode", filename])
                r4 = run_cmd(["pdflatex", "-interaction=nonstopmode", filename])

                if r4.returncode != 0:
                    all_out = "\n\n".join(
                        [
                            f"$ {' '.join(cmd)} (exit {code})\n{out}"
                            for cmd, code, out in logs
                        ]
                    )
                    lines = all_out.splitlines()
                    tail = "\n".join(lines[-40:])
                    self.append_assistant_message(
                        escape_html(
                            "Compilation failed. Tail of logs:\n\n" + tail
                        )
                    )
                    return

            self.append_assistant_message(
                escape_html("Compilation finished. Opening PDF...")
            )
            self.open_thesis_pdf()
        except FileNotFoundError as e:
            self.append_assistant_message(
                escape_html(
                    f"Compilation tool not found ({e}). Install TeXLive/latexmk or ensure tools are in PATH."
                )
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

    # ---------- thesis tools ----------

    def tool_read_thesis_metadata(self) -> Dict[str, Any]:
        if not self.current_tex_path:
            return {"error": "No thesis file loaded."}
        if not self.thesis_sections:
            self.parse_thesis_sections()
        sections = [
            {"index": i, "level": sec["level"], "title": sec["title"]}
            for i, sec in enumerate(self.thesis_sections)
        ]
        return {"path": self.current_tex_path, "sections": sections}

    def tool_read_thesis_section(self, index: int) -> Dict[str, Any]:
        if not self.current_tex_path:
            return {"error": "No thesis file loaded."}
        if not self.thesis_sections:
            self.parse_thesis_sections()
        if index < 0 or index >= len(self.thesis_sections):
            return {"error": f"Invalid index {index}."}
        sec = self.thesis_sections[index]
        start, end = sec["start"], sec["end"]
        text = self.current_tex_content[start:end]
        return {
            "index": index,
            "level": sec["level"],
            "title": sec["title"],
            "text": text,
        }

    def tool_write_thesis_section(self, index: int, new_text: str) -> Dict[str, Any]:
        if not self.current_tex_path:
            return {"error": "No thesis file loaded."}
        if not self.thesis_sections:
            self.parse_thesis_sections()
        if index < 0 or index >= len(self.thesis_sections):
            return {"error": f"Invalid index {index}."}
        sec = self.thesis_sections[index]
        start, end = sec["start"], sec["end"]
        try:
            self.current_tex_content = (
                self.current_tex_content[:start]
                + new_text
                + self.current_tex_content[end:]
            )
            with open(self.current_tex_path, "w", encoding="utf-8") as f:
                f.write(self.current_tex_content)
            old_title = sec["title"]
            old_level = sec["level"]
            self.parse_thesis_sections()
            return {
                "status": "ok",
                "message": f"Replaced {old_level} '{old_title}' in {self.current_tex_path}.",
            }
        except Exception as e:
            return {"error": f"Error writing section: {e}"}

    # ---------- RAG tool ----------

    def tool_search_local_corpus(self, query: str, top_k: int) -> Dict[str, Any]:
        if not self.citations_enabled:
            return {"query": query, "disabled": True, "results": []}
        try:
            results = self.dm.search(query, top_k=top_k)
        except Exception as e:
            return {"query": query, "error": f"search error: {e}", "results": []}

        structured = []
        for i, r in enumerate(results, start=1):
            refs = r.get("references") or {}
            citation_key = refs.get("citation_key") or r.get("citation_key", "")
            title = refs.get("title") or r.get("title")
            snippet = self.get_snippet(r)
            structured.append(
                {
                    "rank": i,
                    "suggested_citation_key": citation_key,
                    "pdf_filename": r.get("pdf_filename", ""),
                    "chunk_filename": r.get("chunk_filename", ""),
                    "title": title,
                    "snippet": snippet,
                }
            )
        return {"query": query, "disabled": False, "results": structured}

    def get_snippet(self, result: dict) -> str:
        chunk_file = result.get("chunk_filename", "")
        if not chunk_file:
            return "No chunk_filename in result."
        chunk_path = os.path.join(self.dm.chunk_files_directory, chunk_file)
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return "Could not read chunk."

    # ---------- BibTeX helpers (read-only) ----------

    def get_bib_path(self) -> Optional[str]:
        """
        Find the thesis BibTeX file.

        Priority:
        1. THESIS_BIB_PATH (if exists)
        2. references.bib or *.bib in directory of current_tex_path
        3. references.bib or *.bib in parent directory of current_tex_path
        4. references.bib or *.bib in THESIS_DIR
        5. references.bib or *.bib in parent of THESIS_DIR
        """
        # 1. explicit override
        if THESIS_BIB_PATH:
            path = os.path.expanduser(THESIS_BIB_PATH)
            if os.path.exists(path):
                return os.path.abspath(path)

        dirs: List[str] = []

        if self.current_tex_path:
            d = os.path.dirname(self.current_tex_path)
            dirs.append(d)
            parent = os.path.dirname(d)
            dirs.append(parent)

        dirs.append(THESIS_DIR)
        dirs.append(os.path.dirname(THESIS_DIR))

        seen = set()
        for d in dirs:
            if not d or d in seen or not os.path.isdir(d):
                continue
            seen.add(d)
            # references.bib first
            cand = os.path.join(d, "references.bib")
            if os.path.exists(cand):
                return os.path.abspath(cand)
            # any .bib
            for fname in os.listdir(d):
                if fname.lower().endswith(".bib"):
                    return os.path.abspath(os.path.join(d, fname))

        return None

    def parse_bib_entries(self) -> (Dict[str, str], Optional[str]):
        bib_path = self.get_bib_path()
        if not bib_path or not os.path.exists(bib_path):
            return {}, bib_path

        if self._bib_cache_path == bib_path and self._bib_cache:
            return self._bib_cache, bib_path

        try:
            with open(bib_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return {}, bib_path

        entries: Dict[str, str] = {}
        entry_start_re = re.compile(r'@\w+\s*\{\s*([^,]+)\s*,')
        current_key = None
        current_lines: List[str] = []

        for line in lines:
            m = entry_start_re.match(line)
            if m:
                if current_key is not None and current_lines:
                    entries[current_key] = "".join(current_lines).strip()
                current_key = m.group(1).strip()
                current_lines = [line]
            else:
                if current_key is not None:
                    current_lines.append(line)

        if current_key is not None and current_lines:
            entries[current_key] = "".join(current_lines).strip()

        self._bib_cache = entries
        self._bib_cache_path = bib_path
        return entries, bib_path

    def tool_read_bib_index(self) -> Dict[str, Any]:
        entries, bib_path = self.parse_bib_entries()
        if not bib_path:
            return {"error": "No BibTeX file found (check THESIS_BIB_PATH or thesis directories)."}
        if not entries:
            return {"path": bib_path, "entries": []}

        index_list = []
        title_re = re.compile(r'(?im)^\s*title\s*=\s*[{"](.+?)[}"],?\s*$', re.MULTILINE)
        for key, entry in entries.items():
            m = title_re.search(entry)
            title = m.group(1).strip() if m else None
            index_list.append({"key": key, "title": title})

        return {"path": bib_path, "entries": index_list}

    def tool_read_bib_entry(self, key: str) -> Dict[str, Any]:
        entries, bib_path = self.parse_bib_entries()
        if not bib_path:
            return {"error": "No BibTeX file found."}
        if key not in entries:
            return {"path": bib_path, "error": f"Key '{key}' not found."}
        return {"path": bib_path, "key": key, "entry": entries[key]}

    # ---------- sessions ----------

    def save_session(self):
        session_data = {"messages": self.messages}
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
        self.set_send_button_pressed(False)
        print(f"Session loaded from {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dm = DataManager()
    window = ReferenceChatGUI(dm)
    window.show()
    sys.exit(app.exec_())
