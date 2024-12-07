from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect, QTimer, QPropertyAnimation, QEasingCurve, QAbstractAnimation,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QPen,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QSizePolicy, QFormLayout, QLineEdit, QPushButton, QVBoxLayout,QScrollArea,QProgressBar,QSpacerItem,
    QWidget, QGraphicsOpacityEffect, QGraphicsDropShadowEffect)

from PySide6 import QtWidgets, QtSvg, QtSvgWidgets
from PySide6.QtSvgWidgets import QSvgWidget
import sys, os


from qfluentwidgets import BodyLabel, ProgressBar



MB = 1048576
GB = 1073741824
class StatusBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        self.p = parent
        
        self.effect = QGraphicsOpacityEffect(self)
        self.effect.setOpacity(1.0)
        self.setGraphicsEffect(self.effect)
        
        self.animation = QPropertyAnimation(self.effect, b"opacity", self)
        self.animation.setStartValue(self.effect.property("opacity"))
        self.animation.setDuration(100)
        
        self.setFixedHeight(30)
        
        self.h = QHBoxLayout()
        self.sp = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.statusText  = BodyLabel(parent=self, text='正在下载')
        
        
        self.progressBar = ProgressBar(self)
        self.progressBar.setValue(0)
        self.progressBar.setFixedHeight(6)
        self.progressBar.setTextVisible(False)
        # self.progressBar.setStyleSheet(
        # '''
        # QProgressBar::chunk {
        #     border-radius:3px;
        #     background-color: #EEEEEE;
        # };
        # background-color: rgba(64, 64, 64, 130);
        # border-color: rgba(255, 255, 255, 0);
        # border-radius:3px;
        # color: rgb(255, 0, 0);
        # '''
        # )
        
        self.h.addWidget(self.progressBar)
        
        # self.h.addSpacerItem(self.sp)
        self.h.addWidget(self.statusText)
        
        
        self.setLayout(self.h)
        
        self.setHidden(True)
        
        self.animation.finished.connect(lambda: self.progressBar.setValue(0))

    def setHidden(self, hidden: bool) -> None:
        
        self.animation.stop()
        self.animation.setStartValue(self.effect.property("opacity"))
        
        if hidden:  
            self.animation.setEndValue(0.)
        else:
            self.animation.setEndValue(1.)
            
        self.animation.start()
            
            
    def _formatBytes(self, bytes:int):
        if bytes < 1024:
            return f'{bytes}B'
        elif bytes < MB:
            return f'{bytes/1024:.2f}KB'
        elif bytes < GB:
            return f'{bytes/MB:.2f}MB'
        else:
            return f'{bytes/GB:.2f}GB'

    def setProgress(self, dbytes:int, totalbytes:int, isBytes=True):
        self.progressBar.setValue(int(dbytes/totalbytes*100))
        if isBytes:
            self.statusText.setText(f' -> {self._formatBytes(dbytes)}/{self._formatBytes(totalbytes)}')
        else:
            self.statusText.setText(f' -> {int(dbytes/totalbytes*100)}%')

    # def paintEvent(self, event) -> None:
        
        
    #     p = QPainter()
    #     brush = QBrush(QColor('#00DDDDDD'))
    #     p.begin(self)
    #     p.setPen(QColor('#00DDDDDD'))
    #     p.setBrush(brush)
        
    #     p.drawRect(self.rect())
        
    #     p.end()
        
    #     # return super().paintEvent(event)
    
