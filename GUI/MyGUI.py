# This file is the GUI used for my demonstration of my trained model

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
import numpy as np
from demoFuncs import predictImage

class DemoGUI(QMainWindow):

    # Constructor for the GUI
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NBA Team Classifier")
        self.setFixedSize(1000, 1000)

        self.uploadButton = QPushButton("Upload Image")
        self.uploadButton.clicked.connect(self.uploadFile)

        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.uploadButton)
        layout.addWidget(self.imageLabel)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # This function is called when the upload button is clicked, and it is responsible for calling the demoFuncs.py file
    def uploadFile(self):
        # Only allow the user to select png, jpg, and jpeg files, ignoring the returned string of filtered file extensions
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.jpeg)")
        
        # If the filepath is not empty, call the predictImage function from demoFuncs.py
        # and display the returned annotated image in the GUI
        if filePath:
            annotatedImg = predictImage(filePath)
            pixmap = QPixmap(annotatedImg)
            self.imageLabel.setPixmap(pixmap.scaled(800, 800, Qt.KeepAspectRatio))

# Load the GUI when the file is run
app = QApplication([])
window = DemoGUI()
window.show()
app.exec_()