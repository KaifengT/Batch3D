import sys, os
import time
from PySide6.QtCore import QTimer, Qt

path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from qfluentwidgets import InfoBarIcon, InfoBar, PushButton, setTheme, Theme, FluentIcon, InfoBarPosition, InfoBarManager


class PopMessageWidget_fluent:
    def __init__(self, parent=None) -> None:
        self.parant = parent
        
        self.map = {
            'error': InfoBarIcon.ERROR,
            'warning': InfoBarIcon.WARNING,
            'complete': InfoBarIcon.SUCCESS,
            'msg': InfoBarIcon.INFORMATION
        }
        
    def add_message_stack(self, msg:tuple=('Null','error')):
        '''
        msg:tuple=(('Title', 'message'),'error/warning/complete/msg')
        '''
        
        
        content = "My name is kira yoshikake, 33 years old. Living in the villa area northeast of duwangting, unmarried. I work in Guiyou chain store. Every day I have to work overtime until 8 p.m. to go home. I don't smoke. The wine is only for a taste. Sleep at 11 p.m. for 8 hours a day. Before I go to bed, I must drink a cup of warm milk, then do 20 minutes of soft exercise, get on the bed, and immediately fall asleep. Never leave fatigue and stress until the next day. Doctors say I'm normal."
        w = InfoBar(
            icon=self.map[msg[1]],
            title=msg[0][0],
            content=msg[0][1],
            orient=Qt.Vertical,    # vertical layout
            isClosable=True,
            position=InfoBarPosition.BOTTOM_RIGHT,
            duration=16000 if msg[1] == 'error' else 8000,
            parent=self.parant
        )
        # w.addWidget(PushButton('Action'))
        w.show()

