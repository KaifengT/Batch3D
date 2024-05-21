
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect)
from PySide6.QtCore import  QPropertyAnimation


class windowBlocker(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        
        self.effect = QGraphicsOpacityEffect(self)
        self.effect.setOpacity(0.0)
        self.setGraphicsEffect(self.effect)
        
        self.animation = QPropertyAnimation(self.effect, b"opacity", self)

        
        self.animation.setStartValue(self.effect.property("opacity"))
        
        self.setObjectName('main_windowBlocker')
        self.setStyleSheet('QFrame#main_windowBlocker{background-color: #101010}')

        self.animation.finished.connect(self.superHidden)
        
        
        super().setHidden(True)


    def setHidden(self, hidden: bool) -> None:
        self.animation.setStartValue(self.effect.property("opacity"))
        if hidden:
            self.animation.setDuration(150)
            self.animation.setEndValue(0)
        else:
            self.animation.setDuration(150)
            self.animation.setEndValue(0.8)
            super().setHidden(False)
        self.animation.start()
        
    def superHidden(self, ):
        # print(self.effect.property("opacity"))
        if self.effect.property("opacity")<0.02:
            return super().setHidden(True)
        else:
            return super().setHidden(False)
            
        
        # return super().setHidden(hidden)        