from PyQt5.QtWidgets import (
    QDialog, QApplication, QLabel, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
import sys
from typing import Optional, Dict


class ChampFichier(QWidget):
    """
    Widget to select a file with a label, a button to open a file dialog,
    and a label displaying the selected file name.
    """
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = QLabel(label)
        self.bouton = QPushButton("Select Calibration images")
        self.chemin_label = QLabel("No file selected")

        self.bouton.clicked.connect(self.open_dialog)

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.bouton)
        layout.addWidget(self.chemin_label)
        self.setLayout(layout)

        self.chemin: str = ""

    def open_dialog(self) -> None:
        """Open a file dialog and update the displayed filename."""
        chemin, _ = QFileDialog.getOpenFileName(self, "Select Calibration images", "")
        if chemin:
            self.chemin = chemin
            self.chemin_label.setText(chemin.split("/")[-1])

    def get_value(self) -> str:
        """Return the selected file path."""
        return self.chemin


class Select_Files(QDialog):
    """
    Dialog containing two ChampFichier widgets for images and video file selection,
    and a button to confirm the selections.
    """
    def __init__(self) -> None:
        super().__init__()
        self.image_field = ChampFichier("Image File")
        self.video_field = ChampFichier("Video File")

        self.bouton_valider = QPushButton("Run Photometric analysis")
        self.bouton_valider.clicked.connect(self.valider)

        layout = QVBoxLayout()
        layout.addWidget(self.image_field)
        layout.addWidget(self.video_field)
        layout.addWidget(self.bouton_valider)

        self.setLayout(layout)
        self.setWindowTitle("Photometric analysis")

        self.resultats: Optional[Dict[str, str]] = None

    def valider(self) -> None:
        """Gather selected file paths and accept the dialog."""
        image_path = self.image_field.get_value()
        video_path = self.video_field.get_value()

        self.resultats = {
            "image_path": image_path,
            "video_path": video_path
        }

        self.accept()


def run() -> Optional[Dict[str, str]]:
    """
    Run the file selection dialog.

    Returns:
        A dictionary with keys "image_path" and "video_path" if dialog accepted,
        otherwise None.
    """
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    dialog = Select_Files()
    if dialog.exec_() == QDialog.Accepted:
        return dialog.resultats
    return None


if __name__ == "__main__":
    selected_files = run()
    if selected_files:
        print(f"Selected images: {selected_files['image_path']}")
        print(f"Selected video: {selected_files['video_path']}")
