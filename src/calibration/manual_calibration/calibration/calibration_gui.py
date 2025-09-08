from datetime import datetime, timezone
from typing import Optional, Dict, Any
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtWidgets import (
    QDialog, QApplication, QLabel, QWidget, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QDateTimeEdit, QComboBox
)
import sys
import os
import json


class ChampFormulaire(QWidget):
    def __init__(self, label: str, valeur_defaut: str = "") -> None:
        super().__init__()
        self.label = QLabel(label)
        self.input = QLineEdit()
        self.input.setText(valeur_defaut)

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        self.setLayout(layout)

    def get_value(self) -> float:
        """Returns the input text converted to float."""
        return float(self.input.text())


class ChampDate(QWidget):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = QLabel(label)
        self.input = QDateTimeEdit()
        self.input.setCalendarPopup(True)
        self.input.setDateTime(QDateTime.currentDateTime())

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        self.setLayout(layout)

    def get_value(self) -> datetime:
        """Returns the selected datetime as a timezone-aware UTC datetime."""
        qdt = self.input.dateTime()
        dt = qdt.toPyDateTime().replace(tzinfo=timezone.utc)
        return dt


class Fenetre(QDialog):
    def __init__(self) -> None:
        super().__init__()

        self.stations: Dict[str, Dict[str, Any]] = self.charger_stations("./utils/stations.json")

        self.combo_station = QComboBox()
        self.combo_station.addItem("")  # Empty default option
        self.combo_station.addItems(self.stations.keys())
        self.combo_station.currentIndexChanged.connect(self.remplir_coordonnees_station)

        self.champ_lat = ChampFormulaire("Latitude", "41.9389")
        self.champ_lon = ChampFormulaire("Longitude", "2.7306")
        self.champ_alt = ChampFormulaire("Altitude", "15")
        self.champ_vfov = ChampFormulaire("Vertical fov", "60")
        self.champ_hfov = ChampFormulaire("Horizontal fov", "100")
        self.champ_az_center = ChampFormulaire("Azimuth center", "325")
        self.champ_alt_center = ChampFormulaire("Altitude center", "30")
        self.champ_date = ChampDate("Date")

        self.bouton_valider = QPushButton("Lancer la calibration")
        self.bouton_valider.clicked.connect(self.valider)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Sélectionner une station :"))
        layout.addWidget(self.combo_station)
        layout.addWidget(self.champ_lat)
        layout.addWidget(self.champ_lon)
        layout.addWidget(self.champ_alt)
        layout.addWidget(self.champ_vfov)
        layout.addWidget(self.champ_hfov)
        layout.addWidget(self.champ_az_center)
        layout.addWidget(self.champ_alt_center)
        layout.addWidget(self.champ_date)
        layout.addWidget(self.bouton_valider)

        self.setLayout(layout)
        self.setWindowTitle("Calibration")

        self.resultats: Optional[Dict[str, Any]] = None

    def charger_stations(self, fichier: str) -> Dict[str, Dict[str, Any]]:
        """Loads station data from JSON file if available."""
        if os.path.exists(fichier):
            with open(fichier, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print(f"Fichier '{fichier}' introuvable. Aucune station chargée.")
            return {}

    def remplir_coordonnees_station(self) -> None:
        """Fills coordinate fields based on selected station."""
        nom = self.combo_station.currentText()
        data = self.stations.get(nom, {})
        if data:
            self.champ_lat.input.setText(str(data.get("latitude", "")))
            self.champ_lon.input.setText(str(data.get("longitude", "")))
            self.champ_alt.input.setText(str(data.get("altitude", "")))

    def valider(self) -> None:
        """Collects input data into a results dictionary and closes the dialog."""
        self.resultats = {
            "lat": self.champ_lat.get_value(),
            "lon": self.champ_lon.get_value(),
            "alt": self.champ_alt.get_value(),
            "vfov": self.champ_vfov.get_value(),
            "hfov": self.champ_hfov.get_value(),
            "az_center": self.champ_az_center.get_value(),
            "alt_center": self.champ_alt_center.get_value(),
            "date": self.champ_date.get_value()
        }
        self.accept()


def run_calibration() -> Optional[Dict[str, Any]]:
    """
    Runs the calibration dialog.

    Returns:
        Optional[Dict[str, Any]]: Dictionary with calibration parameters if validated, else None.
    """
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    dialog = Fenetre()
    if dialog.exec_() == QDialog.Accepted:
        return dialog.resultats
    return None
