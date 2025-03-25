# metadata_editor.py
import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QLabel, QLineEdit, QPushButton, QFormLayout,
    QScrollArea, QDialog, QDialogButtonBox, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from data_manager import DataManager

# A list of common BibTeX fields. Adjust as needed:
BIBTEX_FIELDS = [
    "type",
    "citation_key",
    "title",
    "author",
    "journal",
    "year",
    "volume",
    "number",
    "pages",
    "publisher",
    "address",
    "note",
    "doi",
    "url"
]

class MetadataEditorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()  # Use the same data manager you have in data_manager.py
        self.setWindowTitle("Metadata & References Editor")

        self.pdf_list = QListWidget()
        self.pdf_list.itemClicked.connect(self.on_select_pdf)

        self.form_layout = QFormLayout()
        self.field_inputs = {}  # key: field name, value: QLineEdit

        self.save_button = QPushButton("Save Changes")
        self.save_button.clicked.connect(self.save_changes)
        self.regen_button = QPushButton("Regenerate references.bib")
        self.regen_button.clicked.connect(self.regenerate_bib)

        # Build a scrollable panel for the form fields
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.form_layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(scroll_widget)

        # Layouts
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout.addWidget(QLabel("PDFs in metadata.json:"))
        left_layout.addWidget(self.pdf_list)

        right_layout.addWidget(QLabel("Reference Fields:"))
        right_layout.addWidget(self.scroll_area)
        right_layout.addWidget(self.save_button)
        right_layout.addWidget(self.regen_button)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        # Populate the PDF list
        self.load_pdfs()

    def load_pdfs(self):
        """Populate the QListWidget with all the PDFs from metadata."""
        self.pdf_list.clear()
        for pdf_filename in self.dm.metadata.keys():
            item = QListWidgetItem(pdf_filename)
            self.pdf_list.addItem(item)

    def on_select_pdf(self, item):
        """When the user selects a PDF, display its reference fields in the form."""
        pdf_filename = item.text()
        references = self.dm.metadata[pdf_filename].get("references", {})

        # Clear old fields
        for i in reversed(range(self.form_layout.count())):
            form_item = self.form_layout.itemAt(i)
            if form_item:
                w = form_item.widget()
                if w:
                    w.deleteLater()

        self.field_inputs = {}

        # Show each known BibTeX field as a row
        for field in BIBTEX_FIELDS:
            value = references.get(field, "")
            line_edit = QLineEdit(str(value))
            self.field_inputs[field] = line_edit
            self.form_layout.addRow(QLabel(field + ":"), line_edit)

        # If references has any unknown or custom fields beyond BIBTEX_FIELDS, show them too
        for custom_field, val in references.items():
            if custom_field not in BIBTEX_FIELDS:
                line_edit = QLineEdit(str(val))
                self.field_inputs[custom_field] = line_edit
                self.form_layout.addRow(QLabel(custom_field + ":"), line_edit)

    def save_changes(self):
        """Save the current form’s field values back to the metadata for the selected PDF."""
        current_item = self.pdf_list.currentItem()
        if not current_item:
            return

        pdf_filename = current_item.text()
        if pdf_filename not in self.dm.metadata:
            return

        # Read the form’s data
        new_references = {}
        for field, line_edit in self.field_inputs.items():
            val = line_edit.text().strip()
            if val:  # Only store non-empty fields
                new_references[field] = val

        # Save back to data_manager’s metadata
        self.dm.metadata[pdf_filename]["references"] = new_references
        # Optionally set references_extracted to True, since we have them
        self.dm.metadata[pdf_filename]["references_extracted"] = True

        # Persist to metadata.json
        self.dm.save_metadata()

        # Inform user
        QMessageBox.information(self, "Saved", f"References updated for {pdf_filename}")

    def regenerate_bib(self):
        """Call generate_references_bib() to rebuild references.bib from the updated metadata."""
        self.dm.generate_references_bib()
        QMessageBox.information(self, "Done", "Regenerated references.bib")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = MetadataEditorGUI()
    editor.show()
    sys.exit(app.exec_())
