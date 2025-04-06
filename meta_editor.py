# metadata_editor.py
import sys
import os
import json
import shutil

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QLabel, QLineEdit, QPushButton, QFormLayout,
    QScrollArea, QMessageBox
)
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
        self.dm = DataManager()  # your DataManager
        self.setWindowTitle("Metadata & References Editor")

        # We'll maintain a list of *all* known PDF filenames:
        self.all_pdf_filenames = sorted(self.dm.metadata.keys())

        # Search bar for filtering PDFs
        self.search_label = QLabel("Search PDFs / references:")
        self.search_line = QLineEdit()
        self.search_line.setPlaceholderText("Enter search term...")
        self.search_line.textChanged.connect(self.update_pdf_list_filtered)

        # List of PDFs
        self.pdf_list = QListWidget()
        self.pdf_list.itemClicked.connect(self.on_select_pdf)

        # Form layout on the right for showing / editing reference fields
        self.form_layout = QFormLayout()
        self.field_inputs = {}  # key: field name, value: QLineEdit

        # Buttons
        self.save_button = QPushButton("Save Changes")
        self.save_button.clicked.connect(self.save_changes)

        self.regen_button = QPushButton("Regenerate references.bib")
        self.regen_button.clicked.connect(self.regenerate_bib)

        self.delete_button = QPushButton("Delete PDF Entry")
        self.delete_button.clicked.connect(self.delete_pdf_entry)

        # Build a scrollable panel for the form fields
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.form_layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(scroll_widget)

        # Lay out everything
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Left side: search bar + PDF list
        left_layout.addWidget(self.search_label)
        left_layout.addWidget(self.search_line)
        left_layout.addWidget(QLabel("PDFs in metadata.json:"))
        left_layout.addWidget(self.pdf_list)

        # Right side: reference fields form + buttons
        right_layout.addWidget(QLabel("Reference Fields:"))
        right_layout.addWidget(self.scroll_area)
        right_layout.addWidget(self.save_button)
        right_layout.addWidget(self.regen_button)
        right_layout.addWidget(self.delete_button)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        self.load_pdfs()

    def load_pdfs(self):
        """Initially populate the QListWidget with all the PDFs from metadata."""
        self.update_pdf_list_filtered()

    def update_pdf_list_filtered(self):
        """
        Clears and repopulates self.pdf_list, showing only PDFs that match the search term
        either in their filename or in any reference field.
        """
        query = self.search_line.text().strip().lower()
        self.pdf_list.clear()

        for pdf_filename in self.all_pdf_filenames:
            if pdf_filename not in self.dm.metadata:
                # Might be removed in the meantime
                continue

            # Check filename match
            filename_match = (query in pdf_filename.lower())

            # Check references match
            refs = self.dm.metadata[pdf_filename].get("references", {})
            refs_str = json.dumps(refs, ensure_ascii=False).lower()
            references_match = (query in refs_str)

            if (not query) or filename_match or references_match:
                # If the query is empty, or we match filename or references, list this PDF
                item = QListWidgetItem(pdf_filename)
                self.pdf_list.addItem(item)

    def on_select_pdf(self, item):
        """When the user clicks a PDF, display its reference fields in the form."""
        pdf_filename = item.text()
        references = self.dm.metadata[pdf_filename].get("references", {})

        # Clear old fields
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.field_inputs.clear()

        # Show each known BibTeX field as a row
        for field in BIBTEX_FIELDS:
            value = references.get(field, "")
            line_edit = QLineEdit(str(value))
            self.field_inputs[field] = line_edit
            self.form_layout.addRow(QLabel(field + ":"), line_edit)

        # If references has any unknown or custom fields, show them too
        for custom_field, val in references.items():
            if custom_field not in BIBTEX_FIELDS:
                line_edit = QLineEdit(str(val))
                self.field_inputs[custom_field] = line_edit
                self.form_layout.addRow(QLabel(custom_field + ":"), line_edit)

    def save_changes(self):
        """
        Save the current form’s field values back to the metadata for
        the selected PDF in the list, then persist to metadata.json.
        """
        current_item = self.pdf_list.currentItem()
        if not current_item:
            return

        pdf_filename = current_item.text()
        if pdf_filename not in self.dm.metadata:
            return

        # Gather updated reference data
        new_references = {}
        for field, line_edit in self.field_inputs.items():
            val = line_edit.text().strip()
            if val:
                new_references[field] = val

        # Update DataManager’s metadata
        self.dm.metadata[pdf_filename]["references"] = new_references
        self.dm.metadata[pdf_filename]["references_extracted"] = True

        # Persist to metadata.json
        self.dm.save_metadata()

        # Inform user
        QMessageBox.information(self, "Saved", f"References updated for {pdf_filename}")

        # Optionally refresh the list in case the updated reference changed search results
        self.update_pdf_list_filtered()

    def regenerate_bib(self):
        """
        Call generate_references_bib() to rebuild references.bib from the updated metadata.
        """
        self.dm.generate_references_bib()
        QMessageBox.information(self, "Done", "Regenerated references.bib")

    def delete_pdf_entry(self):
        """
        Remove the selected PDF from metadata, references, embeddings, etc.,
        and move the PDF file to pdfs/unscannable.
        """
        current_item = self.pdf_list.currentItem()
        if not current_item:
            return

        pdf_filename = current_item.text()
        if pdf_filename not in self.dm.metadata:
            return

        # Confirm user wants to delete
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete '{pdf_filename}' from metadata and move its PDF to unscannable?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        # Call a helper method in data_manager (explained below)
        self.dm.remove_pdf_entry(pdf_filename)

        # Also remove from our local list and refresh
        if pdf_filename in self.all_pdf_filenames:
            self.all_pdf_filenames.remove(pdf_filename)

        self.update_pdf_list_filtered()

        QMessageBox.information(self, "Deleted", f"Removed '{pdf_filename}' from metadata.")

        # Optionally, regenerate references immediately
        self.dm.generate_references_bib()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = MetadataEditorGUI()
    editor.show()
    sys.exit(app.exec_())
