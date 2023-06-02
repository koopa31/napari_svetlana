from qtpy.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QPushButton
from qtpy.QtCore import Qt


class DemoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.case = 0
        self.setWindowTitle("Choose the image type")
        b1 = QPushButton("2D")
        b1.clicked.connect(self.set_multi2d)

        b2 = QPushButton("3D")
        b2.clicked.connect(self.set_3D)

        self.buttonBox = QDialogButtonBox(Qt.Vertical)

        self.buttonBox.addButton(b1, QDialogButtonBox.ActionRole)
        self.buttonBox.addButton(b2, QDialogButtonBox.ActionRole)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def set_multi2d(self):
        self.case = "2D"
        self.close()

    def set_3D(self):
        self.case = "3D"
        self.close()

    def get_case(self):
        return self.case
