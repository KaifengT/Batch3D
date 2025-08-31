
from PySide6.QtCore import Qt, Signal, QRect, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath
from PySide6.QtWidgets import QWidget, QVBoxLayout
from qfluentwidgets import BodyLabel, Theme
import os


class DragDropWidget(QWidget):

    filesDropped = Signal(list)
    
    def __init__(self, parent=None, acceptedExtensions=None):
        super().__init__(parent)
        
        self.acceptedExtensions = acceptedExtensions or []
        
        self.is_drag_over = False
        
        self.fade_animation = QPropertyAnimation(self, b"geometry")
        self.fade_animation.setDuration(100)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        self.setMinimumSize(300, 200)
        
        self.border_color = QColor(100, 150, 255)
        self.bg_color = QColor(255, 255, 255, 220)
        self.drag_bg_color = QColor(255, 255, 255, 220)



        self.animationMagnitude = 20
        
        self.setBaseGeometry()
        self._setup_ui()


    def setBaseGeometry(self):

        self.baseGeometry = self.geometry()
        self.centerBig = QRect(self.baseGeometry.x() - self.animationMagnitude / 2, self.baseGeometry.y() - self.animationMagnitude / 2,
                               self.baseGeometry.width() + self.animationMagnitude, self.baseGeometry.height() + self.animationMagnitude)


    def setGeometry(self, *args, **kwargs):
        ret = super().setGeometry(*args, **kwargs)
        self.setBaseGeometry()
        return ret
    
    def setThemeColor(self, color:QColor):
        self.border_color = color
        self.update()


    def setTheme(self, theme:Theme):
        if theme == Theme.DARK:
            color = QColor(30, 30, 30, 220)
        else:
            color = QColor(255, 255, 255, 220)
        self.bg_color = color
        self.drag_bg_color = color
        self.update()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        self.hint_label = BodyLabel("Drag files here")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        
        if self.acceptedExtensions:
            format_text = "Supported: " + ", ".join(self.acceptedExtensions)
        else:
            format_text = "Supports all file formats"

        self.format_label = BodyLabel(format_text)
        self.format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        
        layout.addStretch()
        layout.addWidget(self.hint_label)
        layout.addWidget(self.format_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def enterAnimation(self):
        if not self.is_drag_over:
            self.is_drag_over = True
            self.hint_label.setText("Drop files")

            self.fade_animation.stop()
            self.fade_animation.setStartValue(self.geometry())
            self.fade_animation.setEndValue(self.centerBig)
            self.fade_animation.start()
            
            self.update()
    
    def leaveAnimation(self):
        if self.is_drag_over:
            self.is_drag_over = False
            self.hint_label.setText("Drag files here")

            self.fade_animation.stop()
            self.fade_animation.setStartValue(self.geometry())
            self.fade_animation.setEndValue(self.baseGeometry)
            self.fade_animation.start()
            
            self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        painter.setBrush(QBrush(self.bg_color))
        painter.setPen(QPen(self.border_color, 2, Qt.PenStyle.DashLine))        
        path = QPainterPath()
        path.addRoundedRect(rect.adjusted(5, 5, -5, -5), 10, 10)
        painter.drawPath(path)
        painter.end()
    
    def set_accepted_extensions(self, extensions):
        self.acceptedExtensions = extensions or []
        
        if self.acceptedExtensions:
            format_text = "Supported: " + ", ".join(self.acceptedExtensions)
        else:
            format_text = "Supports all file formats"
        self.format_label.setText(format_text)
