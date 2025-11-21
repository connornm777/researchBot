import sys
import json
from typing import Dict, Any

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QLineEdit,
    QPushButton,
    QFormLayout,
    QScrollArea,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices

from data_manager import DataManager

# Standard BibTeX-ish fields to show first
BIBTEX_FIELDS = [
    "citation_key",
    "type",
    "title",
    "author",
    "year",
    "journal",
    "booktitle",
    "publisher",
    "volume",
    "number",
    "pages",
    "doi",
    "url",
]


class MetadataEditorGUI(QWidget):
    """
    Simple GUI to inspect and edit references in metadata.json.

    Left:  search + PDF list
    Right: BibTeX-style fields for selected PDF + buttons:
           - Save Changes
           - Regenerate references.bib
           - Delete PDF Entry
           - Open PDF
    """

    def __init__(self) -> None:
        super().__init__()

        self.dm = DataManager()
        self.setWindowTitle("Metadata & References Editor")

        self.all_pdf_filenames = sorted(self.dm.metadata.keys())
        self.current_pdf: str | None = None
        self.field_inputs: Dict[str, QLineEdit] = {}

        self._build_ui()
        self.update_pdf_list_filtered()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # left panel: search + list
        left_layout = QVBoxLayout()
        left_layout.setSpacing(4)

        self.search_label = QLabel("Search PDFs / references:")
        self.search_line = QLineEdit()
        self.search_line.setPlaceholderText("Enter search term...")
        self.search_line.textChanged.connect(self.update_pdf_list_filtered)

        self.pdf_list = QListWidget()
        self.pdf_list.itemClicked.connect(self.on_select_pdf)

        left_layout.addWidget(self.search_label)
        left_layout.addWidget(self.search_line)
        left_layout.addWidget(self.pdf_list)

        # right panel: current PDF + form + buttons
        right_layout = QVBoxLayout()
        right_layout.setSpacing(4)

        self.current_pdf_label = QLabel("No PDF selected")
        self.current_pdf_label.setStyleSheet("font-weight: bold;")

        # scrollable form
        self.form_layout = QFormLayout()
        self.form_layout.setSpacing(2)
        form_container = QWidget()
        form_container.setLayout(self.form_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(form_container)

        # buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(4)

        self.save_button = QPushButton("Save Changes")
        self.save_button.clicked.connect(self.save_changes)

        self.regen_button = QPushButton("Regenerate references.bib")
        self.regen_button.clicked.connect(self.regenerate_bib)

        self.delete_button = QPushButton("Delete PDF Entry")
        self.delete_button.clicked.connect(self.delete_pdf_entry)

        self.open_pdf_button = QPushButton("Open PDF")
        self.open_pdf_button.clicked.connect(self.open_pdf)

        button_row.addWidget(self.save_button)
        button_row.addWidget(self.regen_button)
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.open_pdf_button)
        button_row.addStretch(1)

        right_layout.addWidget(self.current_pdf_label)
        right_layout.addWidget(scroll_area, stretch=1)
        right_layout.addLayout(button_row)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

    # ---------- helpers ----------

    def _current_references(self) -> Dict[str, Any]:
        if not self.current_pdf:
            return {}
        return self.dm.metadata.get(self.current_pdf, {}).get("references", {}) or {}

    # ---------- left panel ----------

    def update_pdf_list_filtered(self) -> None:
        query = self.search_line.text().strip().lower()
        self.pdf_list.clear()

        for pdf_filename in self.all_pdf_filenames:
            data = self.dm.metadata.get(pdf_filename, {})
            refs = data.get("references") or {}
            refs_str = json.dumps(refs, ensure_ascii=False).lower()

            filename_match = query in pdf_filename.lower()
            references_match = query in refs_str

            if not query or filename_match or references_match:
                item = QListWidgetItem(pdf_filename)
                self.pdf_list.addItem(item)

    def on_select_pdf(self, item: QListWidgetItem) -> None:
        pdf_filename = item.text()
        self.current_pdf = pdf_filename
        self.current_pdf_label.setText(f"Editing: {pdf_filename}")

        # clear form
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.field_inputs.clear()

        refs = self.dm.metadata[pdf_filename].get("references", {}) or {}

        # standard fields first
        for field in BIBTEX_FIELDS:
            value = refs.get(field, "")
            le = QLineEdit(str(value))
            self.field_inputs[field] = le
            self.form_layout.addRow(QLabel(field + ":"), le)

        # custom fields
        for custom_field, val in refs.items():
            if custom_field in BIBTEX_FIELDS:
                continue
            le = QLineEdit(str(val))
            self.field_inputs[custom_field] = le
            label = QLabel(custom_field + ":")
            label.setStyleSheet("color: #AAAAAA;")  # mark as custom
            self.form_layout.addRow(label, le)

    # ---------- right panel actions ----------

    def save_changes(self) -> None:
        if not self.current_pdf:
            return

        pdf_filename = self.current_pdf
        if pdf_filename not in self.dm.metadata:
            return

        new_refs: Dict[str, Any] = {}
        for field, line_edit in self.field_inputs.items():
            text = line_edit.text().strip()
            if text == "":
                continue
            new_refs[field] = text

        if "citation_key" not in new_refs or not new_refs["citation_key"].strip():
            QMessageBox.warning(
                self,
                "Missing citation_key",
                "Please provide a non-empty 'citation_key' before saving.",
            )
            return

        if "type" not in new_refs or not new_refs["type"]:
            new_refs["type"] = "misc"

        self.dm.metadata[pdf_filename]["references"] = new_refs
        self.dm.metadata[pdf_filename]["references_extracted"] = True
        self.dm.save_metadata()

        QMessageBox.information(
            self,
            "Saved",
            f"References updated for {pdf_filename}",
        )

        self.update_pdf_list_filtered()

    def regenerate_bib(self) -> None:
        self.dm.generate_references_bib()
        QMessageBox.information(self, "Done", "Regenerated references.bib")

    def delete_pdf_entry(self) -> None:
        if not self.current_pdf:
            return

        pdf_filename = self.current_pdf
        if pdf_filename not in self.dm.metadata:
            return

        reply = QMessageBox.question(
            self,
            "Confirm deletion",
            f"Remove '{pdf_filename}' from metadata, move PDF to unscannable,\n"
            "and delete associated chunks/embeddings?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.dm.remove_pdf_entry(pdf_filename)
        self.all_pdf_filenames = sorted(self.dm.metadata.keys())
        self.current_pdf = None
        self.current_pdf_label.setText("No PDF selected")
        self.update_pdf_list_filtered()

        QMessageBox.information(
            self,
            "Deleted",
            f"Removed '{pdf_filename}' and reindexed embeddings.",
        )

    def open_pdf(self) -> None:
        if not self.current_pdf:
            return
        pdf_filename = self.current_pdf
        pdf_path = self.dm.pdf_files_directory / pdf_filename
        if not pdf_path.exists():
            QMessageBox.warning(
                self,
                "Missing file",
                f"PDF file not found:\n{pdf_path}",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf_path)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = MetadataEditorGUI()
    editor.show()
    sys.exit(app.exec_())
