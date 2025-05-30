import sys
import cv2
import serial
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from rfdeon.command.command import Command, CMD_INVENTORY_ALL
from rfdeon.response.response import Response
from rfdeon.response.inventory_all import InventoryAll
from rfdeon.util.parse_util import bytes_to_hex_readable
from rfdeon.util.reader_util import get_response_serial

import face_recognition as fr

ENCODINGS_PATH = Path("output/encodings.isiot")
HISTORY_FILE = "historyy.txt"
RFID_COM_PORT = "COM5"
RFID_BAUDRATE = 57600

def load_encodings():
    if not ENCODINGS_PATH.exists():
        print("Encoding file not found.")
        return None
    with ENCODINGS_PATH.open("rb") as f:
        return pickle.load(f)

def load_rfid_history():
    history = {}
    try:
        with open(HISTORY_FILE, "r") as file:
            for line in file:
                parts = line.strip().split(" - ")
                if len(parts) == 2:
                    tag = parts[0].strip().upper()
                    history[tag] = parts[1].strip()
    except FileNotFoundError:
        print("History file not found.")
    return history

def normalize_tag(tag: str) -> str:
    tag = tag.replace(" ", "").upper()
    return ' '.join(tag[i:i+2] for i in range(0, len(tag), 2))

class FaceRecognitionThread(QThread):
    face_name_signal = pyqtSignal(str, QImage)

    def __init__(self, encodings):
        super().__init__()
        self.encodings = encodings
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = fr.face_locations(rgb_frame)
            face_encodings = fr.face_encodings(rgb_frame, locations)

            name = "Unknown"
            if face_encodings:
                results = fr.compare_faces(self.encodings["encodings"], face_encodings[0])
                names = Counter(n for match, n in zip(results, self.encodings["names"]) if match)
                name = names.most_common(1)[0][0] if names else "Unknown"
                top, right, bottom, left = locations[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            image = self.convert_cv_qt(frame)
            self.face_name_signal.emit(name, image)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def convert_cv_qt(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        return QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)

class RFIDReaderThread(QThread):
    tag_detected = pyqtSignal(str)

    def run(self):
        try:
            ser = serial.Serial(RFID_COM_PORT, baudrate=RFID_BAUDRATE, timeout=2)
            if not ser.is_open:
                ser.open()

            history = load_rfid_history()
            print("[DEBUG] RFID thread running. Waiting for tag...")

            while True:
                ser.write(Command(CMD_INVENTORY_ALL).serialize())
                response_bytes = get_response_serial(ser)

                if not response_bytes:
                    continue  # tidak ada tag, loop lagi

                response = Response(response_bytes)
                inventory_all = InventoryAll(response.data)

                if inventory_all.tags:
                    tag_hex = bytes_to_hex_readable(inventory_all.tags[0])
                    tag_hex = normalize_tag(tag_hex)
                    print(f"[DEBUG] Tag detected: {tag_hex}")
                    name = history.get(tag_hex, "Unknown")
                    print(f"[DEBUG] Name found: {name}")
                    self.tag_detected.emit(name)
        except Exception as e:
            print(f"[ERROR] {e}")
            self.tag_detected.emit(f"Error: {e}")

class SmartDoorSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Door System")
        self.setGeometry(200, 100, 800, 600)

        self.loaded_encodings = load_encodings()
        self.rfid_history = load_rfid_history()

        self.last_face_name = "Unknown"
        self.last_rfid_name = None

        self.init_ui()
        self.start_threads()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setStyleSheet("background-color: #FFF5E1;")
        central_widget.setLayout(main_layout)

        logo_layout = QHBoxLayout()
        logo_left = QLabel()
        logo_left.setPixmap(QPixmap("assets/image.png").scaledToHeight(60, Qt.SmoothTransformation))
        logo_right = QLabel()
        logo_right.setPixmap(QPixmap("assets/telkom.png").scaledToHeight(60, Qt.SmoothTransformation))
        logo_layout.addWidget(logo_left, alignment=Qt.AlignLeft)
        logo_layout.addStretch()
        logo_layout.addWidget(logo_right, alignment=Qt.AlignRight)
        main_layout.addLayout(logo_layout)

        title = QLabel("SMART DOOR SYSTEM")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #333;")
        main_layout.addWidget(title)

        self.image_label = QLabel()
        self.image_label.setFixedSize(400, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #ddd; border: 4px solid red; margin: 10px;")
        face_layout = QHBoxLayout()
        face_layout.addStretch()
        face_layout.addWidget(self.image_label)
        face_layout.addStretch()
        main_layout.addLayout(face_layout)

        self.name_label = QLabel("Name   : Unknown")
        self.status_label = QLabel("Status : Denied")
        for label in [self.name_label, self.status_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("""
                background-color: #333;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 6px;
                margin: 5px 50px;
            """)
            main_layout.addWidget(label)

        main_layout.addStretch()
        self.setCentralWidget(central_widget)

    def start_threads(self):
        self.face_thread = FaceRecognitionThread(self.loaded_encodings)
        self.face_thread.face_name_signal.connect(self.update_face)
        self.face_thread.start()

        self.start_rfid_scan()

    def start_rfid_scan(self):
        self.rfid_thread = RFIDReaderThread()
        self.rfid_thread.tag_detected.connect(self.update_rfid)
        self.rfid_thread.start()

    def update_face(self, name, frame_image):
        self.last_face_name = name
        self.name_label.setText(f"Name   : {name}")
        self.image_label.setPixmap(QPixmap.fromImage(frame_image).scaled(self.image_label.size(), Qt.KeepAspectRatio))
        self.update_status()

    def update_rfid(self, name):
        self.last_rfid_name = name
        self.update_status()

    def update_status(self):
        if self.last_face_name != "Unknown" and self.last_rfid_name:
            if self.last_face_name.lower() == self.last_rfid_name.lower():
                self.status_label.setText("Status : Granted")
            else:
                self.status_label.setText("Status : Denied")
        else:
            self.status_label.setText("Status : Denied")

    def closeEvent(self, event):
        if hasattr(self, 'face_thread'):
            self.face_thread.stop()
        if hasattr(self, 'rfid_thread'):
            self.rfid_thread.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartDoorSystemGUI()
    window.show()
    sys.exit(app.exec_())
