
import math
from PySide6.QtCore import ( QPropertyAnimation, QEasingCurve,QTimer,
    QSize, QTime, Qt, Signal, Slot, QRect, QPoint)
from PySide6.QtGui import (QBrush, QColor, QPainter, QPen)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QFormLayout, QLineEdit, QPushButton,
    QWidget, QGraphicsOpacityEffect, QGraphicsDropShadowEffect)



class GLAddon_circling(QWidget):

    def __init__(self, parent=None,):
        super().__init__(parent)

        self.alpha = 255
        self.circle_size = 18
        self.bg_color = [69, 76, 90, 0]
        # self.setGeometry(QRect(0,0,48,48))
        self.setFixedSize(40,40)
        
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy1)
        self.setMinimumSize(QSize(24, 24))
        self.setMaximumSize(QSize(32, 32))

        self.angle = 0
        self.angles = 0
        self.count = 0
        self.counts = 0
        self.running = False

        self.count_c = True
        self.counts_c = True

        self.frontColor = [221,221,221,self.alpha]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setValue)
        # self.timer.start(20)
        

    def setValue(self,):
        '''
        state:int 0-100
        '''
        # print(self.count)
        self.count += 0.07

        # if not self.running and self.alpha>0:
        #     self.alpha -= 51

        # if self.running and self.alpha<255:
        #     self.alpha += 51

        # if not self.running and self.alpha<=0:
        #     self.timer.stop()


        # self.angle = (math.cos(self.count)+1) * 180
        self.angles = self.count*100

        self.angle = math.sin(self.count)*80+100

        if self.count > 314: self.count = 0

        self.update()

    def reset(self, ):
        self.count = 0
        self.angles = 0
        self.angle = 0
        self.update()


    def start(self,):
        self.running = True
        if not self.timer.isActive():
            # print('timer start')
            self.timer.start(20)

    def stop(self,):
        self.running = False
        self.timer.stop()
        self.reset()


    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(*self.bg_color))
        painter.drawEllipse(0, 0, self.width(), self.height()) #22
        painter.setBrush(QColor('#DDDDDD'))


        # painter.drawPie(QRect(0, 0, self.width(), self.height()),90*16 , -(self.angle) * 16)
        pen = QPen(QColor(*[221,221,221,self.alpha]))
        pen.setWidth(5)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        
        painter.drawArc(5, 5, self.width()-10, self.height()-10, self.angles*16, self.angle * 16)

        painter.end()
     