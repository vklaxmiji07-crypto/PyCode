from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal

class ScrubbableLabel(QLabel):
    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.SizeHorCursor)
        self._value = 0.0
        self._last_mouse_x = 0
        self._is_scrubbing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_scrubbing = True
            self._last_mouse_x = event.globalX()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._is_scrubbing:
            delta = event.globalX() - self._last_mouse_x
            self._value += delta * 0.01  # Sensitivity
            self.setText(f"{self._value:.3f}")
            self.valueChanged.emit(self._value)
            self._last_mouse_x = event.globalX()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_scrubbing = False
            event.accept()

    def setValue(self, value):
        self._value = value
        self.setText(f"{self._value:.3f}")
