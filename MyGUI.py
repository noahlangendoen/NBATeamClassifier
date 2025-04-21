from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
import numpy as np
from demoFuncs import predictImage



class DemoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NBA Team Classifier")
        self.setFixedSize(400, 400)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.uploadFile)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.prediction_label = QLabel("")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 24px; color: black;")

        layout = QVBoxLayout()
        layout.addWidget(self.upload_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.prediction_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def uploadFile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()
            prediction = predictImage(file_path)
            self.prediction_label.setText(prediction)


app = QApplication(sys.argv)
window = DemoGUI()
window.show()
app.exec_()