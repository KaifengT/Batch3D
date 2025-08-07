import sys, os
import time
from PySide6.QtCore import QTimer, Qt

path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from qfluentwidgets import InfoBarIcon, InfoBar, InfoBarPosition


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
        Pop up a message box
        Args:
            msg (tuple) : (('Title', 'message'), type)
                type can be 'error', 'warning', 'complete', 'msg'
        Returns:
            None
        Usage:
            self.PopMessageWidgetObj.add_message_stack((('Error Title', 'This is an error message'), 'error'))
        '''
        
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
        w.show()

