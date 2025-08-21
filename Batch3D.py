'''
copyright: (c) 2025 by KaifengTang
'''
import sys, os
wdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(wdir)
import numpy as np
import numpy.linalg as linalg
from enum import Enum
import copy
from PySide6.QtWidgets import ( QApplication, QMainWindow, QTableWidgetItem, QWidget, QFileDialog, QDialog, QTextEdit, QGraphicsDropShadowEffect, QHBoxLayout, QVBoxLayout, QLabel)
from PySide6.QtCore import  QSize, QThread, Signal, Qt, QPropertyAnimation, QEasingCurve, QPoint, QRect, QObject, QTimer, QEvent
from PySide6.QtGui import QCloseEvent, QIcon, QFont, QAction, QColor, QSurfaceFormat, QTextCursor
from ui.PopMessageWidget import PopMessageWidget_fluent as PopMessageWidget
import multiprocessing
import io
from ui.ui_main_ui import Ui_MainWindow
from ui.ui_remote_ui import Ui_RemoteWidget
import pickle
from backend import backendEngine, backendSFTP
import hashlib
import traceback
from ui.windowBlocker import windowBlocker
import json
import natsort
from glw.GLMesh import *
import trimesh
from trimesh.visual.material import SimpleMaterial, PBRMaterial
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.color import ColorVisuals
from trimesh.scene.transforms import SceneGraph
import importlib
# This is to avoid the error: "ImportError: numpy.core.multiarray failed to import"
import numpy.core.multiarray

if sys.platform == 'win32':
    try:
        from win32mica import ApplyMica, MicaTheme, MicaStyle
    except:
        ...

########################################################################



from qfluentwidgets import (setTheme, Theme, setThemeColor, qconfig, RoundMenu, widgets, ToggleToolButton, Slider, BodyLabel, PushButton, FluentIconBase, LineEdit)
from qfluentwidgets import FluentIcon as FIF


DEFAULT_WORKSPACE = os.getcwd()
TOOL_UI_WIDTH = 350
PROGBAR_HEIGHT = 50
CONSOLE_HEIGHT = 250

class MyFluentIcon(FluentIconBase, Enum):
    """ Custom icons """

    Folder = "Folder"
    File = "File"


    def path(self, theme=Theme.AUTO):
        # getIconColor() return "white" or "black" according to current theme
        return os.path.join(DEFAULT_WORKSPACE, 'ui', 'icons', f'{self.value}.svg')


class cellWidget(QTableWidgetItem):
    def __init__(self, text: str, fullpath='', isRemote=False, isdir=False) -> None:
        
        self.fullpath = fullpath
        self.isRemote = isRemote
        self.isdir = isdir
        return super().__init__(text,)

class cellWidget_toggle(QTableWidgetItem):
    def __init__(self, icon=None) -> None:
        
        self.button = ToggleToolButton(icon)
        return super().__init__()


class RemoteUI(QDialog):
    
    closedSignal = Signal()
    showSiganl = Signal()
    executeSignal = Signal(str, dict)  
    
    def __init__(self, parent:QWidget=None) -> None:
        super().__init__(parent,)
        self.ui = Ui_RemoteWidget()
        self.ui.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # self.setWindowFlags(Qt.FramelessWindowHint)

        self.ui.pushButton_go.setIcon(FIF.RIGHT_ARROW)
        self.ui.pushButton_refresh.setIcon(FIF.SYNC)
                             
        self.ui.pushButton_connect.clicked.connect(self.connectSFTP)
        self.ui.pushButton_openfolder.clicked.connect(self.openFolder)
        
        self.ui.pushButton_cancel.clicked.connect(self.close)
                
        self.ui.tableWidget.cellDoubleClicked.connect(self.chdirSFTP)
        

        self.ui.pushButton_go.clicked.connect(lambda: self.chdirSFTP_path(self.ui.lineEdit_dir.text()))
        self.ui.pushButton_refresh.clicked.connect(lambda: self.chdirSFTP_path(self.ui.lineEdit_dir.text()))

        self.ui.tableWidget.setColumnWidth(0, 270)
        self.ui.tableWidget.setColumnWidth(1, 160)
        self.ui.tableWidget.setColumnWidth(2, 120)

        self.ui.tableWidget.setBorderVisible(True)
        self.ui.tableWidget.setBorderRadius(6)
        
        self.ui.pushButton_openfolder.setDisabled(True)
        self.ui.pushButton_go.setDisabled(True)
        self.ui.pushButton_refresh.setDisabled(True)
        
        self.configPath = os.path.join(DEFAULT_WORKSPACE, 'ssh.config')
        
    def bytestoReadable(self, n):
        symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
        prefix = {}
        for i, s in enumerate(symbols):
            prefix[s] = 1 << (i + 1) * 10
        for s in reversed(symbols):
            if n >= prefix[s]:
                value = float(n) / prefix[s]
                return '%.1f%s' % (value, s)
        return "%sB" % n


    def serverConnected(self, ):
        self.ui.pushButton_openfolder.setDisabled(False)
        self.ui.pushButton_go.setDisabled(False)
        self.ui.pushButton_refresh.setDisabled(False)

        
    def loadSettings(self, ):
        
        try:
            with open(self.configPath, 'r') as f:
                settings = json.load(f)
                
            self.ui.lineEdit_host.setText(settings['host'])
            self.ui.lineEdit_port.setText(settings['port'])
            self.ui.lineEdit_username.setText(settings['username'])
            self.ui.lineEdit_dir.setText(settings['dir'])
        except:
            print('loadSettings error, using default settings')
            traceback.print_exc()

    def saveSettings(self, ):
        

        try:
            settings = {
                'host':self.ui.lineEdit_host.text(),
                'port':self.ui.lineEdit_port.text(),
                'username':self.ui.lineEdit_username.text(),
                'dir':self.ui.lineEdit_dir.text(),
            }
            
            with open(self.configPath, 'w') as f:
                # pickle.dump(settings, f)
                json.dump(settings, f, indent=4)
        except:
            print('saveSettings error')
            traceback.print_exc()

    def connectSFTP(self, ):
        self.executeSignal.emit('connectSFTP', {
            'host':self.ui.lineEdit_host.text(), 
            'port':self.ui.lineEdit_port.text(), 
            'username':self.ui.lineEdit_username.text(), 
            'passwd':self.ui.lineEdit_passwd.text(), 
            'dir':self.ui.lineEdit_dir.text()})
        
    def chdirSFTP(self, row, col):
        if self.ui.tableWidget.item(row, 0).isdir:
            fullpath = self.ui.tableWidget.item(row, 0).fullpath
            self.executeSignal.emit('sftpListDir', {'dir':fullpath, 'isSet':False, 'onlydir':False})


    def chdirSFTP_path(self, path):
        self.executeSignal.emit('sftpListDir', {'dir':path, 'isSet':False, 'onlydir':False})

        
    def openFolder(self, ):
        self.executeSignal.emit('sftpListDir', {'dir':self.ui.lineEdit_dir.text(), 'recursive':(False, True)[self.ui.comboBox.currentIndex()]})
        self.close()
        
    def openFolder_background(self, ):
        print('openFolder_background')
        self.executeSignal.emit('sftpListDir', {'dir':self.ui.lineEdit_dir.text(), 'recursive':(False, True)[self.ui.comboBox.currentIndex()]})
        
    def setFolderContents(self, files_dict:dict, dirname:str):
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.scrollToTop()
        # self.ui.tableWidget.setHorizontalHeaderLabels([dirname])
        self.ui.lineEdit_dir.setText(dirname)

        files_dict = natsort.natsorted(
            files_dict.items(), 
            key=lambda x: str(int(not x[1]['isdir'])) + x[0],
            # alg=natsort.ns.REAL
        )
        files_dict.reverse()
        files_dict = dict(files_dict)


        for k, v in files_dict.items():
            self.ui.tableWidget.insertRow(0)
            
            if v['isdir']:
                event_widget = cellWidget(k, dirname.rstrip('/') + '/' + k, True, True)
                event_widget.setIcon(MyFluentIcon.Folder.qicon())
                size_weight = QTableWidgetItem('--')

            else:
                event_widget = cellWidget(k, dirname.rstrip('/') + '/' + k, True, False)
                event_widget.setIcon(MyFluentIcon.File.qicon())
                size_weight = QTableWidgetItem(self.bytestoReadable(v['size']))

            size_weight.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            mtime_weight = QTableWidgetItem(v['mtime'])
            self.ui.tableWidget.setItem(0, 0, event_widget)
            self.ui.tableWidget.setItem(0, 2, size_weight)
            self.ui.tableWidget.setItem(0, 1, mtime_weight)
            
        self.ui.tableWidget.insertRow(0)
        event_widget = cellWidget('..', os.path.dirname(dirname), True, True)
        self.ui.tableWidget.setItem(0, 0, event_widget)
            
        

            
        
    def closeEvent(self, event: QCloseEvent) -> None:
        self.closedSignal.emit()
        self.saveSettings()
        return super().closeEvent(event)
    
    def showEvent(self, event: QCloseEvent) -> None:
        self.showSiganl.emit()
        
        self.loadSettings()
        return super().showEvent(event)
        

class fileDetailInfoUI(QDialog):
    
    def __init__(self, parent:QWidget=None) -> None:
        super().__init__(parent,)
        if sys.platform == 'win32':
            self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(500, 700)
        self.setWindowTitle('File Contents')
        
        self.verticalLayout = QVBoxLayout(self)


class stdoutRedirector:

    def __init__(self, ):
        self._cache = ''

    def write(self, info:str):
        self._cache += info
        
    def flush(self):
        pass

    def getCache(self):
        return self._cache

    def clear(self):
        self._cache = ''

class consoleUI(QWidget):

    def __init__(self, parent:QWidget=None) -> None:
        super().__init__(parent,)

        # self.resize(500, 700)
        # self.setWindowTitle('Console')
        
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        self.dragArea = QWidget(self)
        self.dragArea.setFixedHeight(6)
        self.dragArea.setMaximumWidth(100)
        self.dragArea.setCursor(Qt.SizeVerCursor)
        self.dragArea.setStyleSheet("""
            QWidget {
                background-color: rgba(128, 128, 128, 100);
                border-radius: 3px;
                
            }
        """)
        self.dragLayout = QHBoxLayout()
        self.dragLayout.setContentsMargins(0, 0, 0, 0)
        self.dragLayout.setSpacing(0)
        
        self.dragging = False
        self.drag_start_y = 0
        self.original_height = 0
        
        self.textbox = widgets.TextBrowser(self)
        self.textbox.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.commandBox = LineEdit(self)
        
        self.dragLayout.addWidget(self.dragArea)
        self.verticalLayout.addLayout(self.dragLayout)
        self.verticalLayout.addWidget(self.textbox)
        self.verticalLayout.addWidget(self.commandBox)
        
        self.commandBox.returnPressed.connect(self._onEnterPressed)
        self.commandBox.setPlaceholderText('Enter command here and press Enter to execute')
        self.commandBox.textChanged.connect(self._onType)
        self._commandBoxLastText = ''

        self._cache = ''
        self._globals = {}
        
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        self.stdout = stdoutRedirector()
        self.stderr = stdoutRedirector()

        sys.stdout = self.stdout
        sys.stderr = self.stderr
        
        self.flush_timer = QTimer(self)
        self.flush_timer.timeout.connect(self.flush)

        font = self.textbox.font()

        if sys.platform == 'win32':
            self.setAttribute(Qt.WA_TranslucentBackground)
            font = QFont(['Consolas', 'Microsoft Yahei UI'], 10, QFont.Weight.Medium)

        elif sys.platform == 'darwin':
            font = QFont(['SF Mono', 'SF Pro Display'], 10, QFont.Weight.Medium)

        else:
            font = QFont(['Consolas', 'Ubuntu Mono'], 10, QFont.Weight.Medium)

        self.textbox.setFont(font)
        self.commandBox.setFont(font)
        
        self.dragArea.installEventFilter(self)
        
        self.command_history = []
        self.history_index = -1

    def eventFilter(self, obj, event:QEvent):
        if obj == self.dragArea:
            if event.type() == event.Type.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.dragging = True
                    self.drag_start_y = event.globalPosition().y()
                    self.original_height = self.height()
                    return True
            elif event.type() == event.Type.MouseMove:
                if self.dragging:
                    # calculate height change
                    delta_y = event.globalPosition().y() - self.drag_start_y
                    new_height = max(100, self.original_height - delta_y)  # min height 100px

                    # get current position
                    current_x = self.x()
                    current_y = self.y()
                    current_width = self.width()

                    # only change height, keep position and width unchanged
                    # since dragging from the top, adjust y position to keep bottom position unchanged
                    new_y = current_y + (self.height() - new_height)
                    
                    self.setGeometry(current_x, new_y, current_width, new_height)
                    
                    return True
            elif event.type() == event.Type.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.dragging = False
                    return True
        return super().eventFilter(obj, event)


    def restore(self, ):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        
    def flush(self):
        self.flushRedirectorOUT()
        self.flushRedirectorERR()

    def flushRedirectorOUT(self):
        out = self.stdout.getCache()
        if len(out):
            if out[-1] in ('\n'):
                out = out[:-1]  # Remove leading newline if exists
            self.textbox.append(out)
        self.stdout.clear()

    def flushRedirectorERR(self):
        out = self.stderr.getCache()
        if len(out):
            if out[-1] in ('\n'):
                out = out[:-1]  # Remove leading newline if exists
            self.textbox.append(out)
        self.stderr.clear()


    def setGlobals(self, globals: dict):
        self._globals = globals

    def showEvent(self, arg__1):
        self.flush_timer.start(100)  # Flush every 100 ms
        return super().showEvent(arg__1)
    
    def closeEvent(self, arg__1):
        self.flush_timer.stop()
        return super().closeEvent(arg__1)

    def execCommand(self, command: str):
        try:
            code = compile(command, '<console>', 'eval')
            result = eval(code, self._globals)
            if result is not None:
                print(result)
        except SyntaxError:
            try:
                code = compile(command, '<console>', 'exec')
                exec(code, self._globals)

            except Exception as e:
                traceback.print_exc()
        except Exception as e:
            traceback.print_exc()
        
    def _onEnterPressed(self):
        command = self.commandBox.text()
        if command.strip():
            self.add2History(command)
            print('→ ', command)
            self.commandBox.clear()
            self.execCommand(command)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.showPreviousCommand()
        elif event.key() == Qt.Key_Down:
            self.showNextCommand()
        else:
            super().keyPressEvent(event)

    def showPreviousCommand(self):
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.commandBox.setText(self.command_history[-(self.history_index + 1)])

    def showNextCommand(self):
        if self.command_history and self.history_index > 0:
            self.history_index -= 1
            self.commandBox.setText(self.command_history[-(self.history_index + 1)])
        elif self.history_index == 0:
            self.history_index = -1
            self.commandBox.clear()

    def add2History(self, command):
        if command and (not self.command_history or command != self.command_history[-1]):
            self.command_history.append(command)
        self.history_index = -1

    def hideEvent(self, event):
        self.flush_timer.stop()
        return super().hideEvent(event)
    
    def _onType(self, text: str):
    
        pairs = {'(': ')', '[': ']', '{': '}', '<': '>', '\'': '\'', '\"': '\"'}
        cursor_pos = self.commandBox.cursorPosition()

        if cursor_pos > 0 and text[cursor_pos - 1] in pairs and len(text) > len(self._commandBoxLastText):
            last_char = text[cursor_pos - 1]
            closing_char = pairs[last_char]
            self.commandBox.blockSignals(True)
            self.commandBox.insert(closing_char)
            self.commandBox.setCursorPosition(cursor_pos)
            self.commandBox.blockSignals(False)
            
        self._commandBoxLastText = text    


class dataParser:
    
    R_YUP_TO_ZUP = np.array([
                                    [1, 0,  0, 0],
                                    [0, 0, -1, 0],
                                    [0, 1,  0, 0],
                                    [0, 0,  0, 1],
                                ], dtype=np.float32)
    
    
    @staticmethod
    def _isHexColorinName(name) -> str:
        if '#' in name:
            name = name.split('#', 2)[1]
            name = name.split('_', 2)[0]
            if len(name) == 6 or len(name) == 8:
                return name
            else:
                return None
        else:
            return None
        
    @staticmethod
    def _isSizeinName(name) -> float:
        if '&' in name:
            size = name.split('&', 2)[1]
            try:
                size = float(size)
                return size
            except:
                return 2
        else:
            return 2
           
    @staticmethod 
    def _decode_HexColor_to_RGB(hexcolor):
        if hexcolor is None:
            return None
        if len(hexcolor) == 6:
            return tuple(int(hexcolor[i:i+2], 16) / 255. for i in (0, 2, 4))
        elif len(hexcolor) == 8:
            return tuple(int(hexcolor[i:i+2], 16) / 255. for i in (0, 2, 4, 6))
        else:
            return (0.9, 0.9, 0.9, 0.9)        
        
    @staticmethod
    def _get_R_between_two_vec(src, dst):
        """ src: (B, 3)
            dst: (B, 3)
            
            dst = R @ src.T
        """
        src = src / (linalg.norm(src, axis=-1, keepdims=True)) + np.array([0, 1e-5, 1e-5])
        dst = dst / linalg.norm(dst, axis=-1, keepdims=True) 
        y_ax = np.cross(src, dst, )
        y_ax = y_ax / (linalg.norm(y_ax, axis=-1, keepdims=True))
        x_src_ax = np.cross(y_ax, src, )
        x_src_ax = x_src_ax / (linalg.norm(x_src_ax, axis=-1, keepdims=True))
        x_dst_ax = np.cross(y_ax, dst, )
        x_dst_ax = x_dst_ax / (linalg.norm(x_dst_ax, axis=-1, keepdims=True))
        f_src = np.concatenate([x_src_ax[..., None], y_ax[..., None], src[..., None]], axis=-1) # (B, 3, 3)
        f_dst = np.concatenate([x_dst_ax[..., None], y_ax[..., None], dst[..., None]], axis=-1) # (B, 3, 3)
        f_src = np.transpose(f_src, [0, 2, 1])
        r = f_dst @ f_src # (B, 3, 3)
        return r
    
    @staticmethod
    def _getArrowfromLine(v:np.ndarray, color:np.ndarray):
        
        
        if hasattr(color, 'ndim') and color.ndim > 1:
            color = color[..., 1, :]
            color = color.repeat(12, axis=0).reshape(-1, color.shape[-1])
                        
        verctor_line = v[:, 1] - v[:, 0]
        verctor_len = np.linalg.norm(verctor_line, axis=-1, keepdims=True)
        vaild_mask = np.where(verctor_len > 1e-7)
        v = v[vaild_mask[0]]
        verctor_line = v[:, 1] - v[:, 0]
        
        nline = np.linalg.norm(verctor_line, axis=-1, keepdims=True)
        
        B = len(verctor_line)
        BR = dataParser._get_R_between_two_vec(np.array([[0, 0, 1]]).repeat(len(verctor_line), axis=0), verctor_line) # (B, 3, 3)
        temp = Arrow.getTemplate(size=0.05) # (12, 3)
        temp = temp[None, ...].repeat(len(verctor_line), axis=0) # (B, 12, 3)
        
        temp *= nline[:, None]
        
        BR = BR[:, None, ...].repeat(12, axis=1)
        BR = BR.reshape(-1, 3, 3) # (B*12, 3, 3)
        temp = temp.reshape(-1, 1, 3)
        temp = BR @ temp.transpose(0, 2, 1)
        T = v[:, 1, :][:,None,:]
        T = T.repeat(12, axis=1).reshape(-1, 3)
        vertex = temp.transpose(0, 2, 1).reshape(-1, 3)
        vertex = vertex + T
        
        return Arrow(vertex=vertex, color=color)

    @staticmethod
    def _rmnan(v):
        v = np.array(v, dtype=np.float32)
        nan_mask = np.logical_not(np.isnan(v))
        rows_with_nan = np.all(nan_mask, axis=1)
        indices = np.where(rows_with_nan)[0]
        return v[indices]
    
    @staticmethod
    def parseArray(k:str, v:np.ndarray, cm:colorManager, arrow=False):
        
        v = np.nan_to_num(v)
        v = np.float32(v)
        
        
        
        # print(k, ':', v.nbytes, 'bytes')
        assert v.nbytes < 1e8, 'array too large, must slice to show'
        
        n_color = dataParser._decode_HexColor_to_RGB(dataParser._isHexColorinName(k))
        user_color = n_color if n_color is not None else cm.get_next_color()
        
        # -------- lines with arrows
        if (len(v.shape) >= 2 and v.shape[-2] == 2 and v.shape[-1] in (3, 6, 7) and 'line' in k): # (..., 2, 3)
            
            if v.shape[-1] in (6, 7):
                user_color = v[..., 3:].reshape(-1, 2, v.shape[-1]-3)
                v = v[..., :3]
        
            
            v = v.reshape(-1, 2, 3)
            lines = Lines(vertex=v, color=user_color)
            
            #-----# Arrow
            if arrow:
                arrow = dataParser._getArrowfromLine(v, user_color)
                obj = UnionObject()
                obj.add(arrow)
                obj.add(lines)
                
            else:
                obj = lines
                
            if hasattr(user_color, 'ndim') and user_color.ndim > 1:
                user_color = colorManager.extract_dominant_colors(user_color, n_colors=3)[0]
            else:
                user_color = (user_color,)

        # -------- bounding box
        elif (len(v.shape) >= 2 and v.shape[-2] == 8 and v.shape[-1] in (3, 6, 7) and 'bbox' in k): # (..., 8, 3)
            
            if v.shape[-1] in (6, 7):
                user_color = v[..., 3:].reshape(-1, 8, v.shape[-1]-3)
                v = v[..., :3]
            
            v = v.reshape(-1, 8, 3)
            obj = BoundingBox(vertex=v, color=user_color)
            
            if hasattr(user_color, 'ndim') and user_color.ndim > 1:
                user_color = colorManager.extract_dominant_colors(user_color, n_colors=3)[0]
            else:
                user_color = (user_color,)

        
        # -------- pointcloud
        elif len(v.shape) >= 2 and v.shape[-1] == 3: # (..., 3)
            v = v.reshape(-1, 3)
            obj = PointCloud(vertex=v, color=user_color, size=dataParser._isSizeinName(k))
            
            user_color = (user_color,)
                
        # -------- pointcloud with point-wise color
        elif len(v.shape) >= 2 and v.shape[-1] in (6, 7): # (..., 6)
            vertex = v[..., :3].reshape(-1, 3)
            if v.shape[-1] == 6:
                color = v[..., 3:6].reshape(-1, 3)
            else:
                color = v[..., 3:7].reshape(-1, 4)
                
            obj = PointCloud(vertex=vertex, color=color, size=dataParser._isSizeinName(k))
            
            user_color, per = colorManager.extract_dominant_colors(color, n_colors=3)
            
            
        # -------- coordinate axis
        elif len(v.shape) >= 2 and v.shape[-1] == 4 and v.shape[-2] == 4: # (..., 4, 4)
            v = v.reshape(-1, 4, 4)
            B = len(v)
            length = 0.3
            mat = v.repeat(3, axis=0)
            line_base = np.array([[length, 0, 0], 
                    [0, length, 0],
                    [0, 0, length],
                    ], dtype=np.float32) # (3, 3)
            line_base = np.tile(line_base, (B, 1))[..., None] # (3, 3) -> (N3, 3, 1)
            # print(line_base.shape)
            # print(v.shape)
            line_trans = np.einsum('bij,bjk->bik', mat[:, :3, :3], line_base)[..., 0] # B 3
            # line_trans += mat[:, :3, 3]
            line_trans = np.concatenate((np.zeros_like(line_trans) + mat[:, :3, 3], line_trans + mat[:, :3, 3]), axis=1)

            color = np.array(
                [
                    [176, 48, 82, 200],
                    [176, 48, 82, 200],
                    [136, 194, 115, 200],
                    [136, 194, 115, 200],
                    [2, 76, 170, 200],
                    [2, 76, 170, 200],
                ]
            ) / 255.

            color = np.tile(color, (B, 1)).reshape(-1, 4)
            line_trans = line_trans.reshape(-1, 3)
            obj = Lines(vertex=line_trans, color=color)

            
            user_color, per = colorManager.extract_dominant_colors(color, n_colors=3)
            
        else:
            return None, k, ((0.9, 0.9, 0.9),), False
            
        return obj, k, obj.mainColors, True

    @staticmethod
    def parseDict(k:str, v:dict):
        assert 'vertex' in v.keys(), 'mesh missing vertex(vertex)'
        assert 'face' in v.keys(), 'mesh missing face(face)'
        if v['vertex'].shape[-1] == 3:
            obj = Mesh(v['vertex'], v['face'])
        elif v['vertex'].shape[-1] in (6, 7):
            obj = Mesh(v['vertex'][..., :3], v['face'], color=v['vertex'][..., 3:])
        elif v['vertex'].shape[-1] == 9:
            obj = Mesh(v['vertex'][..., :3], v['face'], color=v['vertex'][..., 3:6], norm=v['vertex'][..., 6:])
        elif v['vertex'].shape[-1] == 10:
            obj = Mesh(v['vertex'][..., :3], v['face'], color=v['vertex'][..., 3:7], norm=v['vertex'][..., 7:10])
        else:
            assert 'vertex format error'

        return obj, k, obj.mainColors, False
    
    @staticmethod
    def parseTrimesh(k:str, v:trimesh.parent.Geometry3D, cm:colorManager):
        
        if isinstance(v, trimesh.Scene):
            print(f'parse Trimesh.Scene, {len(v.geometry)} meshes found')
            _meshlist = []
            for _k, mesh in v.geometry.items():
                _meshlist.append(mesh)
            v = trimesh.util.concatenate(_meshlist)

            
        
        if isinstance(v, (trimesh.Trimesh)):

            # if hasattr(v, 'scale') and v.scale > 100:
            #     v.apply_scale(1 / v.scale * 10)
            #     self.PopMessageWidgetObj.add_message_stack((('mesh is too large, auto scaled', ''), 'warning'))
            


            if hasattr(v, 'visual') and hasattr(v.visual, 'material'):

                if isinstance(v.visual.material, (SimpleMaterial, PBRMaterial)):
                    tex = v.visual.material
                else:
                    tex = None
            else:
                tex = None

            # vertex_color = v.visual.vertex_colors /255. if isinstance(v.visual, ColorVisuals) else None
            texcoord = v.visual.uv.view(np.ndarray).astype(np.float32) if isinstance(v.visual, TextureVisuals) and hasattr(v.visual, 'uv') and hasattr(v.visual.uv, 'view') else None

            obj = Mesh(v.vertices.view(np.ndarray).astype(np.float32),
                    v.faces.view(np.ndarray).astype(np.int32),
                    # norm=v.face_normals.view(np.ndarray).astype(np.float32),
                    norm=v.vertex_normals.view(np.ndarray).astype(np.float32),
                    color=None,
                    texture=tex,
                    texcoord=texcoord,
                    faceNorm=False
                    )


            main_colors = obj.mainColors
            return obj, k, main_colors, False



        elif isinstance(v, trimesh.PointCloud):
            if hasattr(v, 'colors') and hasattr(v, 'vertices') and len(v.colors.shape) > 1 and v.colors.shape[0] == v.vertices.shape[0]:
                if np.max(v.colors) > 1:
                    colors = v.colors / 255.
                else:
                    colors = v.colors
                array = np.concatenate((v.vertices, colors), axis=-1)
                return dataParser.parseArray(k, array, cm)
            else:
                return dataParser.parseArray(k, np.array(v.vertices), cm)

            # self.add2ObjPropsTable(v, k, adjustable=True)
            
        else:
            raise ValueError(f'unsupported Trimesh object, {v.__class__.__name__}')

    @staticmethod
    def loadNpFile(file) -> dict:

        obj = np.load(file, allow_pickle=True)
            
        if isinstance(obj, dict):
            ...
        elif isinstance(obj, np.lib.npyio.NpzFile):
            obj = dict(obj)

        elif isinstance(obj, np.ndarray):
            obj = {'numpy file': obj}
        else:
            raise ValueError(f'Unknown numpy file type: {type(obj)}')

        return obj

    @staticmethod
    def loadFromAny(fullpath, extName):
        
        
        
        def _trimeshGetTransformChain(graph:SceneGraph, nodeName:str, ptransform=np.eye(4).astype(np.float32)):
            transform, parent = graph.get(nodeName)
            transform = transform @ ptransform
            if parent is None:
                return transform
            else:
                return _trimeshGetTransformChain(graph, parent, transform)
        
        
        if isinstance(fullpath, str) and os.path.isfile(fullpath):
            _extName = os.path.splitext(fullpath)[-1][1:]

            if _extName.lower() in ('npz', 'npy'):
                obj = dataParser.loadNpFile(fullpath)
            elif _extName.lower() in ('obj', 'ply', 'stl', 'pcd', 'glb', 'xyz', 'gltf'):
                baseName = os.path.basename(fullpath)
                fileName = os.path.splitext(baseName)[0]
                tobj = trimesh.load(fullpath, process=False)
                obj = {}
                if isinstance(tobj, trimesh.Scene):
                    # for _k, mesh in tobj.geometry.items():
                    #     if isinstance(mesh, trimesh.parent.Geometry3D):
                    #         obj[f'{_k}'] = mesh
                    for key, mesh in tobj.geometry.items():
                        if isinstance(mesh, trimesh.parent.Geometry3D):
                            nodeName = tobj.graph.geometry_nodes[key][0]
                            transform = tobj.graph.get(nodeName)[0]
                            if _extName.lower() in ('glb', 'gltf'):
                                transform =dataParser.R_YUP_TO_ZUP @ transform
                            mesh.apply_transform(transform)
                            obj[f'{key}'] = mesh

                elif isinstance(tobj, trimesh.parent.Geometry3D):
                    obj[fileName] = tobj
                    
            elif _extName in ['h5', 'H5']:
                # obj = h5py.File(fullpath, 'r', track_order=True)
                raise NotImplementedError('HDF5 file loading is not supported')
            else:
                obj = pickle.load(open(fullpath, 'rb'))

        # load file from API or slice
        elif isinstance(fullpath, (dict)):
            obj = fullpath

        # load file from remote 
        elif isinstance(fullpath, (io.BytesIO, io.BufferedReader, io.BufferedWriter)):
            if extName in ['npz', 'npy', 'NPY', 'NPZ',]:
                obj = dataParser.loadNpFile(fullpath)
            else:
                obj = pickle.load(fullpath)

        else:
            raise ValueError(f'Unknown file type: {type(fullpath)}')
        
        
        if not isinstance(obj, dict):
            raise RuntimeError('data must be a dict')
                        
        return obj




DEFAULT_SIZE = 3

class App(QMainWindow):
    # ----------------------------------------------------------------------
    sendCodeSignal = Signal(str, str)
    sftpSignal = Signal(str, dict)
    quitBackendSignal = Signal()
    sftpDownloadCancelSignal = Signal()
    
    workspaceUpdatedSignal = Signal(dict)

    def __init__(self,):
        """"""
        super().__init__()
        
        
        # fmt = QSurfaceFormat()
        # fmt.setVersion(3, 3)
        # fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        # fmt.setSamples(4) # Anti-aliasing
        # QSurfaceFormat.setDefaultFormat(fmt)

        
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.tgtTheme = Theme.LIGHT

        # self.toolframe = QFrame(self.ui.openGLWidget)
        # self.ui.tool.setFixedSize(200, 200)
        
        self.ui.tool.setFixedWidth(TOOL_UI_WIDTH)
        self.ui.tool.setMinimumHeight(TOOL_UI_WIDTH)
        self.ui.tool.setParent(self)
        self.ui.tool.move(15, 10)
        # add shadow
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(30)
        self.shadow.setColor(QColor(80, 80, 80))
        self.shadow.setOffset(0, 0)

        self.ui.tool.setGraphicsEffect(self.shadow)
        
        
        self.tool_anim = QPropertyAnimation(self.ui.tool, b"pos")
        self.tool_anim.setDuration(200)
        self.tool_anim.setEasingCurve(QEasingCurve.InOutQuart)

        self.tool_anim_button = ToggleToolButton(FIF.PIN, self)
        self.tool_anim_button.move(380, 15)
        self.tool_anim_button.setFixedSize(30, 30)
        self.tool_anim_button.toggle()
        self.tool_anim_button.toggled.connect(self.moveToolWidget)

        self.tool_b_anim = QPropertyAnimation(self.tool_anim_button, b"pos")
        self.tool_b_anim.setDuration(200)
        self.tool_b_anim.setEasingCurve(QEasingCurve.InOutQuart)
        
        self.PopMessageWidgetObj = PopMessageWidget(self)
        self.windowBlocker = windowBlocker(self) 
        
        self.colormanager = colorManager()
        
        self.resize(1600,900)
        
        self.currentPath = './'
        
        self.isNolyBw = True
        
        self.currentScriptPath = ''
        
        self._workspace_obj = {}
        
        self.obj_properties = {}

        self.ui.tableWidget_obj.setColumnWidth(0, 55)
        self.ui.tableWidget_obj.setColumnWidth(1, 150)
        
        self.ui.tableWidget_obj.setBorderVisible(True)
        self.ui.tableWidget_obj.setBorderRadius(6)

        self.ui.tableWidget.setBorderVisible(True)
        self.ui.tableWidget.setBorderRadius(6)

        self.ui.pushButton_openfolder.clicked.connect(self.openFolder)
        self.ui.pushButton_openfolder.setIcon(FIF.FOLDER)
        self.ui.pushButton_openremotefolder.setIcon(FIF.CLOUD_DOWNLOAD)
        self.ui.tableWidget.currentCellChanged.connect(self.cellClickedCallback)
        self.ui.tableWidget.cellDoubleClicked.connect(self.cellClickedCallback)

        self.ui.pushButton_openscript.clicked.connect(self.openScript)
        self.ui.pushButton_openscript.setIcon(FIF.CODE)
        self.ui.pushButton_runscript.clicked.connect(self.runScript)
        

        # self.backendEngine = backendEngine()
        # self.backend = QThread(self, )
        # self.backendEngine.moveToThread(self.backend)


        self.backendSFTP = backendSFTP()
        self.backendSFTPThread = QThread(self, )
        self.backendSFTP.moveToThread(self.backendSFTPThread)
        self.backendSFTP.executeSignal.connect(self.backendExeUICallback)
        
        # NOTE BUGS infoSignal
        
        # self.backendSFTP.infoSignal.connect(lambda data: QWidget().show())
        
        
        
        # self.backendSFTP.infoSignal.connect(lambda msg: print(msg))
        self.sftpDownloadCancelSignal.connect(self.backendSFTP.cancelDownload)

        self.remoteUI = RemoteUI()
        self.remoteUI.executeSignal.connect(self.backendSFTP.run)
        self.sftpSignal.connect(self.backendSFTP.run)
        
        
        self.fileDetailUI = fileDetailInfoUI()
        self.ui.label_info.setParent(self.fileDetailUI)
        self.fileDetailUI.verticalLayout.addWidget(self.ui.label_info)
        self.ui.pushButton_opendetail.clicked.connect(self.openDetailUI)
        self.ui.pushButton_opendetail.setIcon(FIF.INFO)

        # self.backendEngine.executeGLSignal.connect(self.backendExeGLCallback)
        # self.backendEngine.executeUISignal.connect(self.backendExeUICallback)
        # self.backendEngine.infoSignal.connect(self.PopMessageWidgetObj.add_message_stack)
        # self.sendCodeSignal.connect(self.backendEngine.run)
        
        # self.backendEngine.started.connect(self.runScriptStateChangeRunning)
        # self.backendEngine.finished.connect(self.runScriptStateChangeFinish)
        # self.quitBackendSignal.connect(self.backendEngine.quitLoop)

        # self.backend.start()
        self.backendSFTPThread.start()

        self.ui.pushButton_openconsole.clicked.connect(self.showConsole)
        self.ui.pushButton_openconsole.setIcon(FIF.COMMAND_PROMPT)
        self.console = consoleUI(self)
        self.console.setHidden(True)
        self.console.setGeometry(350, 600, 800, 200)

        self.ui.checkBox_arrow.setOnText('On')
        self.ui.checkBox_arrow.setOffText('Off')

        self.center_all = None
        
        self.ui.pushButton_openremotefolder.clicked.connect(self.openRemoteUI)
        
        self.remoteUI.closedSignal.connect(lambda:self.windowBlocker.setHidden(True))
        self.remoteUI.showSiganl.connect(lambda:self.windowBlocker.setHidden(False))
        self.backendSFTP.listFolderContextSignal.connect(self.remoteUI.setFolderContents)
        
        self.ui.pushButton_runscript.setIcon(FIF.SEND)
        self.ui.pushButton_runscript.setEnabled(False)
        
        self.themeToolButtonMenu = RoundMenu(parent=self)
        self.themeLIGHTAction = QAction(FIF.BRIGHTNESS.icon(), 'Light')
        self.themeDARKAction = QAction(FIF.QUIET_HOURS.icon(), 'Dark')
        # self.themeAUTOAction = QAction(FIF.CONSTRACT.icon(), 'Auto')
        self.themeToolButtonMenu.addAction(self.themeLIGHTAction)
        self.themeToolButtonMenu.addAction(self.themeDARKAction)
        # self.themeToolButtonMenu.addAction(self.themeAUTOAction)
        self.ui.toolButton_theme.setMenu(self.themeToolButtonMenu)
        self.ui.toolButton_theme.setIcon(FIF.PALETTE)
        
        
        self.themeLIGHTAction.triggered.connect(lambda:self.changeTheme(Theme.LIGHT))
        self.themeDARKAction.triggered.connect(lambda:self.changeTheme(Theme.DARK))
        # self.themeAUTOAction.triggered.connect(lambda:self.changeTheme(Theme.AUTO))
        
        self.ui.spinBox.valueChanged.connect(self.slicefromBatch)
        
        
        
        # self.ui.tableWidget.menu = RoundMenu(parent=self.ui.tableWidget)
        # self.ui.tableWidget.menu.addAction(Action(FIF.SYNC, '刷新', triggered=lambda: self.remoteUI.openFolder_background()))
        # self.ui.tableWidget.contextMenuEvent = lambda event: self.ui.tableWidget.menu.exec_(event.globalPos())

        self.configPath = os.path.join(DEFAULT_WORKSPACE, 'user.config')
        self.loadSettings()
        self.changeTheme(self.tgtTheme)
        # self.changeTXTTheme(self.tgtTheme)
        
        # self.setUpWatchForThemeChange()
        
        # rename
        self.GL = self.ui.openGLWidget
        self.resetScriptNamespace()
        self.scriptModules = set()
        self.console.setGlobals(self.script_namespace)
        
        self.lastScriptPaths = ''
    
    def resetScriptNamespace(self, ):
        # delete objects in script namespace
        try:
            if hasattr(self, 'script_namespace'):
                for k, v in self.script_namespace.items():
                    if k in ('Batch3D', 'b3d'):
                        continue
                    if isinstance(v, QThread):
                        v.quit()
                        v.wait()
                    elif isinstance(v, QObject):
                        if hasattr(v, 'isWindow') and v.isWindow():
                            v.close()
                        if hasattr(v, 'deleteLater'):
                            v.deleteLater()
                    else:
                        del v
        except:
            traceback.print_exc()
                
        
        finally:
            self.script_namespace = {'Batch3D':self, 'b3d':self}
    
    def moveToolWidget(self, hide=True):
        self.tool_anim.stop()
        self.tool_b_anim.stop()
        self.tool_anim.setStartValue(self.ui.tool.pos())
        self.tool_b_anim.setStartValue(self.tool_anim_button.pos())
        if not hide:
            self.tool_anim.setEndValue(QPoint(-400, self.ui.tool.pos().y()))
            self.tool_b_anim.setEndValue(QPoint(15, self.tool_anim_button.pos().y()))
        else:
            self.tool_anim.setEndValue(QPoint(20, self.ui.tool.pos().y()))
            self.tool_b_anim.setEndValue(QPoint(350, self.tool_anim_button.pos().y()))
        self.tool_anim.start()
        self.tool_b_anim.start()
        self.update()
        
    def openFolder(self, path=None):
        
        if path is not None and isinstance(path, str):
            self.currentPath = path
        else:
            if os.path.exists(self.currentPath) and os.path.isdir(self.currentPath):
                self.currentPath = QFileDialog.getExistingDirectory(self,"Select Folder",self.currentPath) # 起始路径
            else:
                self.currentPath = QFileDialog.getExistingDirectory(self,"Select Folder",'./') # 起始路径
        
        # print(self.currentPath)
        if len(self.currentPath) and os.path.exists(self.currentPath):
            filelist = os.listdir(self.currentPath)
            filelist = natsort.natsorted(filelist, )
            filelist = filelist[::-1]
            self.ui.tableWidget.setRowCount(0)
            for f in filelist:
                
                self.addFiletoTable(f)
                
    def openRemoteFolder(self, filelist:dict, dirname:str):
        
        filelist = natsort.natsorted(
            filelist.items(), 
            key=lambda x: x[0],
        )
        filelist.reverse()
        filelist = dict(filelist)


        self.ui.tableWidget.setRowCount(0)

        for k, v in filelist.items():
            self.addFiletoTable(k, isRemote=True)

        self.currentPath = DEFAULT_WORKSPACE
        # filelist = filelist[::-1]
        # for f in filelist:
        #     self.addFiletoTable(f, isRemote=True)

    def addFiletoTable(self, filepath:str, isRemote=False):
        extName = os.path.splitext(filepath)[-1][1:]

        if extName.lower() in ['obj', 'pkl', 'cug', 'npy', 'npz', 'ply', 'stl', 'pcd', 'glb', 'xyz'] and not filepath.startswith('.'):

            self.ui.tableWidget.insertRow(0)
            event_widget = cellWidget(filepath, os.path.join(self.currentPath, filepath), isRemote=isRemote)
            event_widget.setIcon(MyFluentIcon.File.qicon())

            self.ui.tableWidget.setItem(0, 0, event_widget)

    def resetObjPropsTable(self, keys:str|list|tuple=None):
        row_count = self.ui.tableWidget_obj.rowCount()
        for i in range(row_count):
            item = self.ui.tableWidget_obj.cellWidget(i, 1)
            key_ = item.text()
            if keys is not None:
                if isinstance(keys, str) and key_ == keys or \
                    isinstance(keys, (tuple, list)) and key_ in keys:
                    item.needsRemove = True
            else:
                item.needsRemove = True
                
    def clearObjPropsTable(self, ):
        row_count = self.ui.tableWidget_obj.rowCount()
        for i in range(row_count)[::-1]:
            item = self.ui.tableWidget_obj.cellWidget(i, 1)
            if item.needsRemove:
                self.ui.tableWidget_obj.removeRow(i)

    def changeObjectProps(self, ):
        row_count = self.ui.tableWidget_obj.rowCount()
        for i in range(row_count):
            key = self.ui.tableWidget_obj.cellWidget(i, 1).text()
            isShow = self.ui.tableWidget_obj.cellWidget(i, 0).isChecked()
            if isShow:
                self.ui.tableWidget_obj.cellWidget(i, 0).setIcon(FIF.VIEW)
            else:
                self.ui.tableWidget_obj.cellWidget(i, 0).setIcon(FIF.HIDE)
            if self.ui.tableWidget_obj.cellWidget(i, 2) is not None:
                size = self.ui.tableWidget_obj.cellWidget(i, 2).value()
            else:
                size = DEFAULT_SIZE
            props = {
                'isShow':isShow,
                'size':size,
            }
            self.ui.openGLWidget.setObjectProps(key, props)
            
    def setObjectProps(self, key, props:dict):
        try:
            if key in self._workspace_obj.keys() and \
                key in self.ui.openGLWidget.objectList.keys():
                    # valid object
                
            
                self.ui.openGLWidget.setObjectProps(key, props)
                for i in range(self.ui.tableWidget_obj.rowCount()):
                    item = self.ui.tableWidget_obj.cellWidget(i, 1)
                    if item.text() == key:
                        if 'isShow' in props.keys():
                            tb = self.ui.tableWidget_obj.cellWidget(i, 0)
                            tb.setChecked(props['isShow'])
                            tb.setIcon(FIF.VIEW if props['isShow'] else FIF.HIDE)
                        if 'size' in props.keys():
                            if self.ui.tableWidget_obj.cellWidget(i, 2) is not None:
                                self.ui.tableWidget_obj.cellWidget(i, 2).setValue(props['size'])
                            else:
                                raise ValueError(f'current objeect {key} do not support "size" key')
                        else:
                            raise ValueError(f'Object {key} props must contain "isShow" or "size" keys')
                        return
                    
        except Exception as e:
            print(f'Error setting object properties for {key}: {e}')
            self.PopMessageWidgetObj.add_message_stack(((f'Error setting object properties for {key}', f'{e}'), 'error'))

    def add2ObjPropsTable(self, obj, name:str, colors=None, adjustable=False):
        
        def add_slider(row_num):
            sl = Slider(Qt.Orientation.Horizontal, None)
            sl.setMaximumSize(96, 24)
            sl.setMaximum(20)
            sl.setSingleStep(2)
            sl.setMinimum(1)
            sl.setValue(DEFAULT_SIZE)
            sl.valueChanged.connect(self.changeObjectProps)
            self.ui.tableWidget_obj.setCellWidget(row_num, 2, sl)
            
        def parseColors2grad(colors:tuple|np.ndarray):
            head = 'qlineargradient(x1:0, y1:0, x2:1, y2:0'
            end = ')'
            total = len(colors)
            if total <=1 : total=2
            for i in range(len(colors)):
                color = [int(c*255) for c in colors[i]]
                head += f', stop:{i/(total-1):.2f} rgb({color[0]}, {color[1]}, {color[2]})'
            return head + end

        color_str = parseColors2grad(colors)
        first_color = colors[0]
        r, g, b = first_color[0]*255, first_color[1]*255, first_color[2]*255
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        font_color = 'black' if luminance > 135 else 'white'
        
            
        styleSheet = f'''
            background-color: {color_str};
            color: {font_color};
            border-radius: 6px;
            padding: 2px;
            margin: 6px;
            '''
        
        # print('add2ObjPropsTable:', name)
        row_count = self.ui.tableWidget_obj.rowCount()

        for i in range(row_count):
            item = self.ui.tableWidget_obj.cellWidget(i, 1)
            
            if item.text() == name:
                # print('find exist name:', name)
                item.needsRemove = False
                item.setStyleSheet(styleSheet)
                if adjustable:
                    if self.ui.tableWidget_obj.cellWidget(i, 2) is None:
                        add_slider(i)
                else:
                    if self.ui.tableWidget_obj.cellWidget(i, 2) is not None:
                        self.ui.tableWidget_obj.removeCellWidget(i, 2)
                
                return
            

        self.ui.tableWidget_obj.insertRow(row_count)

        tt = QLabel(name)
        tt.setStyleSheet(styleSheet)
        tt.setFont(font)

        tt.needsRemove = False
        self.ui.tableWidget_obj.setCellWidget(row_count, 1, tt)

        tb = ToggleToolButton(FIF.VIEW)
        tb.setChecked(True)
        tb.toggled.connect(self.changeObjectProps)
        tb.setMaximumSize(24, 24)
        
        self.ui.tableWidget_obj.setCellWidget(row_count, 0, tb)
        if adjustable:
            add_slider(row_count)
            
    def formatContentInfo(self, obj):
        # textInfo = ' FILE CONTENT: '.center(50, '-') + '\n\n'
        textInfo = '### FILE CONTENT: ' + '\n\n --- \n\n'
        
            
        def create_table_from_dict(d:dict):
            try:
                text = ''
                text += '| | name | type | shape | type |\n'
                text += '| --- | --- | --- | --- | --- |\n'
                for i, (k, v) in enumerate(d.items()):
                    if hasattr(v, 'shape'):
                        text += f'| {i+1} | {k} | {v.__class__.__name__} | {v.shape} | {v.dtype} |\n'
                    else:
                        text += f'| {i+1} | {k} | {v.__class__.__name__} | | | |\n'
                return text + '\n *** \n\n'
            except:
                return 'ERROR'      
        
            
        try:
            if isinstance(obj, dict):
                
                textInfo += create_table_from_dict(obj)
                
                for i, (k, v) in enumerate(obj.items()):
                    textInfo += f'#### {i+1}. '
                    if isinstance(v, np.ndarray):
                        textInfo += f'{k} : ndarray{v.shape}\n\n'
                        textInfo +='```\n'+repr(v) + '\n\n```\n\n'
                        
                    elif isinstance(v, (dict)):
                        textInfo += f'{k} : {v.__class__.__name__}\n\n'
                        for kk, vv in v.items():
                            textInfo += f'\t|-{kk} : {vv.__class__.__name__}\n'
                    
                    else:
                        textInfo += f'{k} : {v.__class__.__name__}\n\n'
                        # textInfo +='      '+repr(v).replace('array([', '').replace('])', '') + '\n\n'
                        textInfo +='```\n'+repr(v) + '\n\n```\n\n'
                        
                        
                        
                    textInfo += '\n' + '-' * 50 + '\n\n'
                

            elif isinstance(obj, trimesh.parent.Geometry3D):
                #TODO
                textInfo += f'#### {obj.__class__.__name__} :\n\n'
                if hasattr(obj, 'vertices'):
                    textInfo += f'vertices : {obj.vertices.shape}\n\n'
                if hasattr(obj, 'faces'):
                    textInfo += f'faces    : {obj.faces.shape}\n\n'
                if hasattr(obj, 'vertex_normals'):
                    textInfo += f'vertex_normals : {obj.vertex_normals.shape}\n\n'
                if hasattr(obj, 'face_normals'):
                    textInfo += f'face_normals : {obj.face_normals.shape}\n\n'
                if hasattr(obj, 'colors'):
                    textInfo += f'colors : {obj.colors.shape}\n\n'
                if hasattr(obj, 'metadata'):
                    for kk, vv in obj.metadata.items():
                        textInfo += f'\t|-{kk} : {vv}\n'
            else:
                textInfo += obj.__class__.__name__ + '\n\n'
        except:
            textInfo += 'ERROR'
        finally:
            
            # with open('./test.md', 'w') as f:
            #     f.write(textInfo)
            
            return textInfo
        
    def _isHexColorinName(self, name) -> str:
        if '#' in name:
            name = name.split('#', 2)[1]
            name = name.split('_', 2)[0]
            if len(name) == 6 or len(name) == 8:
                return name
            else:
                return None
        else:
            return None
        
    def _isSizeinName(self, name) -> float:
        if '&' in name:
            size = name.split('&', 2)[1]
            try:
                size = float(size)
                return size
            except:
                return 2
        else:
            return 2
            
    def _decode_HexColor_to_RGB(self, hexcolor):
        if hexcolor is None:
            return None
        if len(hexcolor) == 6:
            return tuple(int(hexcolor[i:i+2], 16) / 255. for i in (0, 2, 4))
        elif len(hexcolor) == 8:
            return tuple(int(hexcolor[i:i+2], 16) / 255. for i in (0, 2, 4, 6))
        else:
            return (0.9, 0.9, 0.9, 0.9)        
        
    def cellClickedCallback(self, row, col, prow=None, pcol=None):
        try:
            fullpath = self.ui.tableWidget.item(row, col).fullpath
        except:
            return
        isRemote = self.ui.tableWidget.item(row, col).isRemote
        filepath = self.ui.tableWidget.item(row, col).text()
        
        if isRemote:
            self.sftpSignal.emit('downloadFile', {'filename':filepath})
            
        else:
            self.loadObj(fullpath)
            
    def sftpDownloadCancel(self, ):
        self.sftpDownloadCancelSignal.emit()
            

    def setWorkspaceObj(self, obj):
        self._workspace_obj = obj
        maxBatch = 0
        if isinstance(obj, dict):
            for k, v in self._workspace_obj.items():
                if self.isSliceable(v):
                    maxBatch = max(maxBatch, v.shape[0])
                    
        else:
            raise RuntimeError('setWorkspaceObj(obj): obj must be a dict')
        
        self.ui.spinBox.setDisabled(maxBatch == 0)
        self.ui.spinBox.setMaximum(maxBatch-1)
        
        self.workspaceUpdatedSignal.emit(obj)
        
    def isSliceable(self, obj):
        if hasattr(obj, 'shape') and len(obj.shape) > 2:
            return True
        else:
            return False

    def slicefromBatch(self, batch,):
        
        try:
            if self._workspace_obj is not None:
                
                if isinstance(self._workspace_obj, dict):
                
                    if batch >= 0:
                        sliced = {}
                        
                        for k, v in self._workspace_obj.items():
                            if self.isSliceable(v):
                                _max_batch = v.shape[0]
                                op_batch = min(batch, _max_batch-1)
                                sliced[k] = v[op_batch:op_batch+1]
                            else:
                                sliced[k] = v
                    
                        self.loadObj(sliced, setWorkspace=False)
                        
                    else:
                        self.loadObj(self._workspace_obj, setWorkspace=False)
                        
                else:
                    raise RuntimeError('slicefromBatch(batch): _workspace_obj must be a dict')
                        
        
        except:
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.PopMessageWidgetObj.add_message_stack((('slice error occurred', str(exc_value)), 'error'))

    def resetSliceFunc(self, ):
        self.ui.spinBox.valueChanged.disconnect(self.slicefromBatch)
        self.ui.spinBox.setValue(-1)
        self.ui.spinBox.setDisabled(True)
        self.ui.spinBox.setMaximum(-1)
        self.ui.spinBox.setMinimum(-1)
        self.ui.spinBox.valueChanged.connect(self.slicefromBatch)
        
        
    def addObj(self, data:dict):
        if not isinstance(data, dict):
            raise RuntimeError('addObj(data): data must be a dict')
        
        obj = self.getWorkspaceObj()
        obj.update(data)
        self.loadObj_update(obj, keys=list(data.keys()))
        
    def add(self, data:dict):
        self.addObj(data)
        
      
    def updateObj(self, data:dict):
        if not isinstance(data, dict):
            raise RuntimeError('addObj(data): data must be a dict')
        self.loadObj(data)

        
    def rmObj(self, key:str|list[str]):
        """Remove object from OpenGLWidget"""
        if isinstance(key, str):
            key = [key, ]
            
        obj = self.getWorkspaceObj()
        for k in key:
            if k in obj.keys():
                obj[k] = None

        self.loadObj_update(obj, keys=key)

    def rm(self, key:str|list[str]):
        """Remove object from OpenGLWidget"""
        self.rmObj(key=key)
        
    def getWorkspaceObj(self) -> dict:
        """Get the current workspace object"""
        return copy.copy(self._workspace_obj)
        
        
    def clear(self, ):
        self.ui.openGLWidget.reset()
        self.resetObjPropsTable()
        self.clearObjPropsTable()
        self.resetSliceFunc()
        self._workspace_obj = {}
        self.ui.label_info.setText('')
        
      
                
    def loadObj(self, fullpath:str|dict, extName='', setWorkspace=True):
        
        
        self.resetObjPropsTable()
        self.colormanager.reset()
        self.ui.openGLWidget.reset()
        
        
        try:
            
            # load file from multi source
            obj = dataParser.loadFromAny(fullpath, extName)
                        
            # store raw obj to workspace_obj
            if setWorkspace:
                self.resetSliceFunc()
                self.setWorkspaceObj(obj)

            info = self.formatContentInfo(obj)
            self.ui.label_info.setMarkdown(info)
            
            
            if isinstance(obj, dict):
                
                for k, v in obj.items():
                    k = str(k)
                            
                    if isinstance(v, dict):
                        _v, _k, _c, _isadj = dataParser.parseDict(k, v)
                        
                    elif isinstance(v, (trimesh.parent.Geometry3D)):
                        _v, _k, _c, _isadj = dataParser.parseTrimesh(k, v, cm=self.colormanager)
                    
                    elif hasattr(v, 'shape'):
                        _v, _k, _c, _isadj = dataParser.parseArray(k, v, cm=self.colormanager, arrow=self.ui.checkBox_arrow.isChecked())

                        
                    if _v:
                        self.ui.openGLWidget.updateObject(ID=_k, obj=_v)
                        self.add2ObjPropsTable(_v, _k, _c, _isadj)
                    
            else:
                raise ValueError(f'Unsupported object type: {type(obj)}')
            
        except:
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.PopMessageWidgetObj.add_message_stack((('file load error', str(exc_value)), 'error'))
            
        self.clearObjPropsTable()
        self.changeObjectProps()
        
        
    def loadObj_update(self, fullpath:str|dict, keys:list, extName='', setWorkspace=True):
        
        
        self.resetObjPropsTable(keys=keys)
        _delete = []
        try:
            
            # load file from multi source
            obj = dataParser.loadFromAny(fullpath, extName)
                        


            
            
            if isinstance(obj, dict):
                
                for k, v in obj.items():
                    k = str(k)
                    if k not in keys:
                        continue
                    
                    # store deleted keys
                    if v is None:
                        self.ui.openGLWidget.updateObject(ID=k, obj=None)
                        _delete.append(k)
                        continue
                            
                    if isinstance(v, dict):
                        _v, _k, _c, _isadj = dataParser.parseDict(k, v)
                        
                    elif isinstance(v, (trimesh.parent.Geometry3D)):
                        _v, _k, _c, _isadj = dataParser.parseTrimesh(k, v, cm=self.colormanager)
                    
                    elif hasattr(v, 'shape'):
                        _v, _k, _c, _isadj = dataParser.parseArray(k, v, cm=self.colormanager, arrow=self.ui.checkBox_arrow.isChecked())

                        
                    if _v:
                        self.ui.openGLWidget.updateObject(ID=_k, obj=_v)
                        self.add2ObjPropsTable(_v, _k, _c, _isadj)
                    
            else:
                raise ValueError(f'Unsupported object type: {type(obj)}')
            
            # delete keys
            for k in _delete:
                if k in obj.keys():
                    obj.pop(k)

            info = self.formatContentInfo(obj)
            self.ui.label_info.setMarkdown(info)

            # store raw obj to workspace_obj
            if setWorkspace:
                self.resetSliceFunc()
                self.setWorkspaceObj(obj)

            
        except:
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.PopMessageWidgetObj.add_message_stack((('file load error', str(exc_value)), 'error'))
            
        self.clearObjPropsTable()
        self.changeObjectProps()
        
        
        
        
        
    def setObjTransform(self, ID, transform:np.ndarray=None):
        """
        Set the transformation matrix for an object in the OpenGL widget.
        
        Parameters:
        - ID: The identifier for the object.
        - transform: A 4x4 transformation matrix.
        """
        
        self.ui.openGLWidget.setObjTransform(ID, transform)
        self.ui.openGLWidget.update()


    def backendExeGLCallback(self, func, kwargs):
        getattr(self.ui.openGLWidget, func)(**kwargs)
        
    def backendExeUICallback(self, func, kwargs):
        getattr(self, func)(**kwargs)
    

    def openScript(self, fullpath=None):
        if fullpath:
            currentScriptPath = fullpath
        else:
            currentScriptPath = QFileDialog.getOpenFileName(self,"Select Script",self.currentPath, '*.py')[0] # 起始路径

        # print(self.currentScriptPath)

        if len(currentScriptPath) and os.path.isfile(currentScriptPath):
            self.ui.pushButton_runscript.setEnabled(True)
            self.currentScriptPath = currentScriptPath
            self.ui.pushButton_runscript.setText('Run [ ' + os.path.basename(self.currentScriptPath) + ' ]')
            
    def runScript(self, ):
        
        if not hasattr(self, 'sysModules'):
            # This should only be done once
            self.sysModules = set(sys.modules.keys())

        self.resetScriptNamespace()
        os.chdir(DEFAULT_WORKSPACE)
        # print(f'Current working directory: {os.getcwd()}')

        if self.lastScriptPaths in sys.path:
            print(f'Remove last script path: {self.lastScriptPaths}')
            sys.path.remove(self.lastScriptPaths)

        if os.path.isfile(self.currentScriptPath):
            
            fname = os.path.basename(self.currentScriptPath)
            scriptFolder = os.path.dirname(self.currentScriptPath)
            os.chdir(scriptFolder)
            sys.path.append(scriptFolder)
            self.lastScriptPaths = scriptFolder
            
            print(f'Current working directory: {os.getcwd()}')
            
            with open(self.currentScriptPath, encoding='utf-8') as f:

                code = f.read()
                code = code.replace('import Batch3D', '') # Deprecated

            try:
                self.console.setGlobals(self.script_namespace)
                
                code = compile(code, fname, 'exec')
                # exec(code, self._globals)
                
                for scriptmodule in self.scriptModules:
                    if scriptmodule in sys.modules.keys():
                        if hasattr(sys.modules[scriptmodule], '__file__'):
                            modulePath = sys.modules[scriptmodule].__file__
                            if isinstance(modulePath, str):
                                if (DEFAULT_WORKSPACE in modulePath) or ('site-packages' in modulePath):
                                    print(f"Pass for sys module: {scriptmodule} from {modulePath}")
                                    ...
                                else:
                                    # print(f"Try to delete script module: {scriptmodule} from {modulePath}")
                                    del sys.modules[scriptmodule]
                                    # if os.path.exists(modulePath):
                                    #     importlib.reload(sys.modules[scriptmodule])
                                    # else:
                                    #     ...
                                    #     print(f"Try to reload Module {scriptmodule} from {modulePath} but it does not exist.")
                        else:
                            ...
                            # print(f"Try to sys module: {scriptmodule} but it does not have a file path: {sys.modules[scriptmodule]}")

                print(f" --- Executing script: {fname} .... --- ")
                exec(code, self.script_namespace)
                
                print(f" --- Executing script: {fname} done --- ")

                self.scriptModules = set(sys.modules.keys()) - self.sysModules

                if sys.platform == 'win32':
                    for item_name, item_instance in self.script_namespace.items():
                        if isinstance(item_instance, QWidget) and item_instance.isWindow():
                            hwnd = item_instance.winId()
                            if hwnd != 0:
                                self.applyMicaTheme(hwnd)
                            
                            item_instance.setStyleSheet("background-color: #00000000;")
                
            except Exception as e:
                exc_type, exc_value, exc_tb = sys.exc_info()
                
                # 准备详细的错误信息
                error_details_list = traceback.format_exception(exc_type, exc_value, exc_tb)
                error_details_str = "".join(error_details_list)
                
                # 1. 在控制台打印详细的错误信息和追溯
                print(f" --- Error executing script: {fname} --- ")
                print(error_details_str)
                print(f" --- End of script error for: {fname} --- ")
                
                # 2. 在UI的PopMessageWidget中显示错误摘要
                self.PopMessageWidgetObj.add_message_stack(
                    ((f"Script '{fname}' exec error", str(exc_value)), 'error')
                )
                
                # NOTE
                # os.chdir(DEFAULT_WORKSPACE)


    # def runScriptStateChangeRunning(self, ):
    #     self.ui.pushButton_runscript.setText('Terminate Script')
    #     # self.ui.pushButton_runscript.applyStyleSheet(**Button_Style_R)
    #     self.ui.pushButton_runscript.setIcon(FIF.CANCEL)
    #     self.ui.pushButton_runscript.disconnect(self)
    #     self.ui.pushButton_runscript.clicked.connect(self.runScriptTerminate)
    #     self.ui.widget_circle.start()
        
    # def runScriptStateChangeFinish(self, ):
    #     # print('nb')
    #     self.ui.pushButton_runscript.setText('Run Script')
    #     self.ui.pushButton_runscript.setIcon(FIF.SEND)
    #     # self.ui.pushButton_runscript.applyStyleSheet(**Button_Style_G)
    #     self.ui.pushButton_runscript.disconnect(self)
    #     self.ui.pushButton_runscript.clicked.connect(self.runScript)
    #     self.ui.widget_circle.stop()
        
    # def runScriptTerminate(self, ):
    #     self.quitBackendSignal.emit()
    #     self.backend.quit()
    #     self.backend.wait()
        
    #     self.runScriptStateChangeFinish()
    #     self.backend.start()

    def getFilePathFromList(self, row:int):
        fullpath = self.ui.tableWidget.item(row, 0).fullpath
        isRemote = self.ui.tableWidget.item(row, 0).isRemote

        return fullpath, isRemote

    def getListLength(self, ):
        return self.ui.tableWidget.rowCount()

    def popMessage(self, title:str='', message:str='', mtype='msg'):
        self.PopMessageWidgetObj.add_message_stack(((title, message), mtype))



    def closeEvent(self, event: QCloseEvent) -> None:
        
        self.saveSettings()
        
        # self.backend.quit()
        # self.backend.wait(200)
        # self.backend.terminate()
        # 
        self.backendSFTPThread.quit()
        self.backendSFTPThread.wait(200)
        self.backendSFTPThread.terminate()
        
        self.resetScriptNamespace()
        self.fileDetailUI.close()
        self.remoteUI.close()

        self.console.restore()
        
        return super().closeEvent(event)
    

    def applyMicaTheme(self, winId):
        if sys.platform == 'win32':
            try:
                m = {
                    Theme.LIGHT:MicaTheme.LIGHT,
                    Theme.DARK:MicaTheme.DARK,
                }

                ApplyMica(winId, m[self.tgtTheme], MicaStyle.DEFAULT)
            except:
                print(f'ApplyMica id {winId} failed')


    def openRemoteUI(self, ):
        self.applyMicaTheme(self.remoteUI.winId())
        self.remoteUI.show()
        
    def openDetailUI(self, ):
        self.fileDetailUI.close()
        self.applyMicaTheme(self.fileDetailUI.winId())
        self.fileDetailUI.show()
        
    def setDownloadProgress(self, dbytes:int, totalbytes:int, isBytes=True):
        self.ui.openGLWidget.statusbar.setProgress(dbytes, totalbytes, isBytes)
        
    def setDownloadProgressHidden(self, hidden:bool):
        self.ui.openGLWidget.statusbar.setHidden(hidden)
        
    def resizeEvent(self, event: QCloseEvent) -> None:
        self.windowBlocker.resize(self.size())   

        self.ui.tool.setFixedWidth(TOOL_UI_WIDTH)
        self.ui.tool.setFixedHeight(self.height() - PROGBAR_HEIGHT)


        current_height = self.console.height()
        self.console.setGeometry(TOOL_UI_WIDTH+30, self.height()-current_height-PROGBAR_HEIGHT+5, self.width()-TOOL_UI_WIDTH-45, min(current_height, self.height()-PROGBAR_HEIGHT))

        return super().resizeEvent(event)

    def serverConnected(self, ):
        self.remoteUI.serverConnected()
        
    # def setUpWatchForThemeChange(self, ):
    #     self.watchForThemeTimer = QTimer()
    #     self.watchForThemeTimer.timeout.connect(self.watchForThemeChange)
    #     self.watchForThemeTimer.start(500)
    #     self.watchForThemeTimer.setSingleShot(False)
        # self.currentTheme = qconfig.theme
    
    def watchForThemeChange(self, ):
        global CURRENT_THEME
        if CURRENT_THEME != qconfig.theme and self.tgtTheme == Theme.AUTO:
            
            setTheme(CURRENT_THEME)
            
    def changeTXTTheme(self, theme):
        global CURRENT_THEME
        if theme == Theme.LIGHT:
            label_info_color = '#202020'
            tool_color = '#FEFEFE'
            shadow_color = '#808080'
        elif theme == Theme.DARK:
            label_info_color = '#FEFEFE'
            tool_color = '#201e1c'
            shadow_color = '#101010'
        
        self.ui.label_info.setStyleSheet(
            '''
            QTextBrowser
                {{
                    background-color: #00000000;
                    
                    border-radius: 6px;
                    border: 0px;
                    font: 1000 8pt;
                    
                    color: {0};
                }}
            '''.format(label_info_color)
        )
        
        self.ui.tool.setStyleSheet(
            '''
            QWidget
            {{
            background-color:{0};
            border-radius:10px;
            }}
            '''.format(tool_color)
        )
        
        self.shadow.setColor(QColor(shadow_color))
        self.ui.tool.setGraphicsEffect(self.shadow)
        self.update()
            
    def changeTheme(self, theme):
        global CURRENT_THEME
        self.tgtTheme = theme

        self.changeTXTTheme(theme)

        setTheme(theme)
        
        self.saveSettings()
        
        if sys.platform == 'win32':
            self.applyMicaTheme(self.remoteUI.winId())
            self.applyMicaTheme(self.fileDetailUI.winId())
            self.applyMicaTheme(self.winId())
            
            
    def loadSettings(self, ):
        
        try:
            with open(self.configPath, 'r') as f:
                settings = json.load(f)
                
                m = {
                    'Light':Theme.LIGHT,
                    'Dark':Theme.DARK,
                    'Auto':Theme.AUTO
                }
                
                if not isinstance(settings, dict):
                    settings = {}
                
                self.tgtTheme = m[settings.get('theme', 'Light')]
                

                self.ui.openGLWidget.gl_settings.setSettings(settings.get('gl_settings', {}))

                self.currentScriptPath = settings.get('lastScript', '')
                if len(self.currentScriptPath) and os.path.isfile(self.currentScriptPath):
                    self.ui.pushButton_runscript.setEnabled(True)
                    self.ui.pushButton_runscript.setText('Run [ ' + os.path.basename(self.currentScriptPath) + ' ]')

                self.openFolder(settings.get('localPath', DEFAULT_WORKSPACE))

        except:
            traceback.print_exc()

    def saveSettings(self, ):
        
        try:
            settings = {
                'theme':self.tgtTheme.value,
                'camera':self.ui.openGLWidget.gl_camera_control_combobox.currentRouteKey(),
                'lastScript':self.currentScriptPath,
                'localPath': self.currentPath,
                'gl_settings': self.ui.openGLWidget.gl_settings.getSettings()
            }
            
            with open(self.configPath, 'w') as f:
                
                json.dump(settings, f, indent=4)
        except:
            traceback.print_exc()


    def dragEnterEvent(self, event):
        
        event.accept()
        
    def dropEvent(self, event):
        
        file = event.mimeData().urls()[0].toLocalFile()
        if os.path.isfile(file):
            folderPath = os.path.dirname(file)
            baseName = os.path.basename(file)
            fileName, ext = os.path.splitext(baseName)
            if ext.lower() in ('.py', '.txt'):
                self.openScript(file)
                self.runScript()
            else:
                self.openFolder(folderPath)
                for i in range(self.ui.tableWidget.rowCount()):
                    item = self.ui.tableWidget.item(i, 0)
                    if os.path.basename(item.fullpath) == baseName:
                        self.ui.tableWidget.setCurrentItem(item)
                        break
                self.loadObj(file)
        else:
            self.openFolder(file)
            
            
    def showConsole(self):
        if self.console.isHidden():
            # self.console.setGeometry(TOOL_UI_WIDTH+30, self.height()-CONSOLE_HEIGHT-PROGBAR_HEIGHT+5, self.width()-TOOL_UI_WIDTH-45, CONSOLE_HEIGHT)
            self.console.setHidden(False)
        else:
            self.console.setHidden(True)


def changeGlobalTheme(x):
    global CURRENT_THEME
    CURRENT_THEME = [Theme.LIGHT, Theme.DARK][x]
    
def setup_opengl_format():
    """设置OpenGL格式"""
    format = QSurfaceFormat()
    format.setVersion(3, 3)
    format.setProfile(QSurfaceFormat.CompatibilityProfile)
    format.setSamples(4)  # 4x MSAA
    format.setDepthBufferSize(24)
    format.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(format)

if __name__ == "__main__":
    
    CURRENT_THEME = qconfig.theme
    
    try:
        from tools.getWinColor import get_windows_colorization_color
        win_colorization_color = get_windows_colorization_color()
        if win_colorization_color is not None:
            setThemeColor(f'#{win_colorization_color:06x}')
    except:
        ...
    
    
    
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)

    # setup_opengl_format()

    App = App()
    
    App.setWindowTitle('Batch3D Viewer build 1.8')
    App.setWindowIcon(QIcon('icon.ico'))
    

    App.remoteUI.setWindowIcon(QIcon('icon.ico'))
    App.fileDetailUI.setWindowIcon(QIcon('icon.ico'))


    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
        try:
            App.setAttribute(Qt.WA_TranslucentBackground)
            font = QFont([u'Microsoft Yahei UI'], 10)
            app.setFont(font)

        except:
            traceback.print_exc()

    elif sys.platform == 'darwin':
        font = QFont(['SF Pro Display', 'Helvetica Neue', 'Arial'], 10, QFont.Weight.Normal)
        app.setFont(font)
        
    else:
        font = QFont(['Ubuntu', 'Arial'], 10, QFont.Weight.Normal)
        app.setFont(font)
    
    
    App.show()

    app.exec()
    
    