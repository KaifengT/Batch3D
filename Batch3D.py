'''
copyright: (c) 2024 by KaifengTang
'''
import sys, os
wdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(wdir)
import numpy as np
import numpy.linalg as linalg
from enum import Enum
from PySide6.QtWidgets import ( QApplication, QMainWindow, QTableWidgetItem, QWidget, QFileDialog, QDialog, QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QFrame, QVBoxLayout, QLabel)
from PySide6.QtCore import  QSize, QThread, Signal, Qt, QPropertyAnimation, QEasingCurve, QPoint, QRect, QObject
from PySide6.QtGui import QCloseEvent, QIcon, QFont, QAction, QColor, QSurfaceFormat
from ui.PopMessageWidget import PopMessageWidget_fluent as PopMessageWidget
import multiprocessing
import io
from ui.addon import GLAddon_ind
from ui.ui_main_ui import Ui_MainWindow
from ui.ui_remote_ui import Ui_RemoteWidget
import pickle
from backend import backendEngine, backendSFTP

import traceback
from ui.windowBlocker import windowBlocker
import json
import natsort
from glw.mesh import *
import trimesh
import h5py

if sys.platform == 'win32':
    try:
        from win32mica import ApplyMica, MicaTheme, MicaStyle
    except:
        ...

########################################################################



from qfluentwidgets import (setTheme, Theme, setThemeColor, qconfig, RoundMenu, widgets, ToggleToolButton, Slider, Action, PushButton, FluentIconBase)
from qfluentwidgets import FluentIcon as FIF


class MyFluentIcon(FluentIconBase, Enum):
    """ Custom icons """

    Folder = "Folder"
    File = "File"


    def path(self, theme=Theme.AUTO):
        # getIconColor() return "white" or "black" according to current theme
        return f'ui/icons/{self.value}.svg'


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
        
        self.configPath = './ssh.config'
        
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
            with open(self.configPath, 'rb') as f:
                settings = pickle.load(f)
                
            self.ui.lineEdit_host.setText(settings['host'])
            self.ui.lineEdit_port.setText(settings['port'])
            self.ui.lineEdit_username.setText(settings['username'])
            self.ui.lineEdit_passwd.setText(settings['passwd'])
            self.ui.lineEdit_dir.setText(settings['dir'])
        except:
            ...

    def saveSettings(self, ):
        
        try:
            settings = {
                'host':self.ui.lineEdit_host.text(),
                'port':self.ui.lineEdit_port.text(),
                'username':self.ui.lineEdit_username.text(),
                'passwd':self.ui.lineEdit_passwd.text(),
                'dir':self.ui.lineEdit_dir.text(),
            }
            
            with open(self.configPath, 'wb') as f:
                pickle.dump(settings, f)
                # json.dump(settings, f, indent=4)
        except:
            ...
        
        
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


DEFAULT_SIZE = 3

class App(QMainWindow):
    # ----------------------------------------------------------------------
    sendCodeSignal = Signal(str, str)
    sftpSignal = Signal(str, dict)
    quitBackendSignal = Signal()

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
        
        self.ui.tool.setFixedWidth(320)
        self.ui.tool.setMinimumHeight(320)
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
        self.tool_anim_button.move(350, 15)
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
        
        self.workspace_obj = None
        
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
        

        self.backendEngine = backendEngine()
        self.backend = QThread(self, )
        self.backendEngine.moveToThread(self.backend)


        self.backendSFTP = backendSFTP()
        self.backendSFTPThread = QThread(self, )
        self.backendSFTP.moveToThread(self.backendSFTPThread)
        self.backendSFTP.executeSignal.connect(self.backendExeUICallback)
        self.backendSFTP.infoSignal.connect(self.PopMessageWidgetObj.add_message_stack)

        self.remoteUI = RemoteUI()
        self.remoteUI.executeSignal.connect(self.backendSFTP.run)
        self.sftpSignal.connect(self.backendSFTP.run)
        
        self.fileDetailUI = fileDetailInfoUI()
        self.ui.label_info.setParent(self.fileDetailUI)
        self.fileDetailUI.verticalLayout.addWidget(self.ui.label_info)
        self.ui.pushButton_opendetail.clicked.connect(self.openDetailUI)
        self.ui.pushButton_opendetail.setIcon(FIF.INFO)

        self.backendEngine.executeGLSignal.connect(self.backendExeGLCallback)
        self.backendEngine.executeUISignal.connect(self.backendExeUICallback)
        self.backendEngine.infoSignal.connect(self.PopMessageWidgetObj.add_message_stack)
        self.sendCodeSignal.connect(self.backendEngine.run)
        
        self.backendEngine.started.connect(self.runScriptStateChangeRunning)
        self.backendEngine.finished.connect(self.runScriptStateChangeFinish)
        self.quitBackendSignal.connect(self.backendEngine.quitLoop)

        self.backend.start()
        self.backendSFTPThread.start()

        self.isTrackObject = False
        
        self.ui.checkBox_arrow.setOnText('On')
        self.ui.checkBox_arrow.setOffText('Off')

        self.center_all = None
        
        self.ui.pushButton_openremotefolder.clicked.connect(self.openRemoteUI)
        
        self.remoteUI.closedSignal.connect(lambda:self.windowBlocker.setHidden(True))
        self.remoteUI.showSiganl.connect(lambda:self.windowBlocker.setHidden(False))
        self.backendSFTP.listFolderContextSignal.connect(self.remoteUI.setFolderContents)
        
        self.ui.pushButton_runscript.setIcon(FIF.SEND)
        
        
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


        
        
        self.configPath = './user.config'
        self.loadSettings()
        self.changeTheme(self.tgtTheme)
        # self.changeTXTTheme(self.tgtTheme)
        
        # self.setUpWatchForThemeChange()
        
        # rename
        self.GL = self.ui.openGLWidget
        self.reset_script_namespace()
    
    def reset_script_namespace(self, ):
        # delete objects in script namespace
        try:
            if hasattr(self, 'script_namespace'):
                for k, v in self.script_namespace.items():
                    if k == 'Batch3D':
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
            self.script_namespace = {'Batch3D':self,}
    
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


        # filelist = filelist[::-1]
        # for f in filelist:
        #     self.addFiletoTable(f, isRemote=True)

    def addFiletoTable(self, filepath:str, isRemote=False):
        extName = os.path.splitext(filepath)[-1][1:]
        
        if extName in ['obj', 'pkl', 'cug', 'npy', 'npz', 'OBJ', 'PKL', 'CUG', 'NPY', 'NPZ', 'ply', 'stl', 'pcd', 'glb', 'xyz', 'PLY', 'STL', 'PCD', 'XYZ', 'GLB', 'h5', 'H5'] and not filepath.startswith('.'):
        
            self.ui.tableWidget.insertRow(0)
            event_widget = cellWidget(filepath, os.path.join(self.currentPath, filepath), isRemote=isRemote)
            event_widget.setIcon(MyFluentIcon.File.qicon())

            self.ui.tableWidget.setItem(0, 0, event_widget)

    def resetObjPropsTable(self, ):
        row_count = self.ui.tableWidget_obj.rowCount()
        for i in range(row_count):
            item = self.ui.tableWidget_obj.cellWidget(i, 1)
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
                
            elif isinstance(obj, h5py.File):
                #TODO
                ...
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

    def setWorkspaceObj(self, obj):
        self.workspace_obj = obj
        maxBatch = 999999999
        if isinstance(obj, dict):
            for k, v in self.workspace_obj.items():
                
                if self.isSliceable(v):
                    maxBatch = min(maxBatch, v.shape[0])
                    
        elif isinstance(obj, h5py.File):
            maxBatch = len(obj)
        
        self.ui.spinBox.setDisabled(maxBatch == 0)
        self.ui.spinBox.setMaximum(maxBatch-1)
        
    def isSliceable(self, obj):
        if hasattr(obj, 'shape') and len(obj.shape) > 2:
            return True
        elif isinstance(obj, h5py.File) and len(obj) > 1:
            return True
        else:
            return False

    def slicefromBatch(self, batch,):
        
        try:
            if self.workspace_obj is not None:
                
                if isinstance(self.workspace_obj, dict):
                
                    if batch >= 0:
                        sliced = {}
                        
                        for k, v in self.workspace_obj.items():
                            if self.isSliceable(v):
                                sliced[k] = v[batch:batch+1]
                            else:
                                sliced[k] = v
                    
                        # print('Sliced:')
                        # for k, v in sliced.items():
                        #     print(k, ':', v.shape)
                    
                        self.loadObj(sliced)
                        
                    else:
                        self.loadObj(self.workspace_obj)
                        
                elif isinstance(self.workspace_obj, h5py.File):
                    if batch < 0:
                        batch = 0
                        
                    sliced = {}
                    
                    group_names = list(self.workspace_obj.keys())
                    
                    this_object = self.workspace_obj[group_names[batch]]
                    
                    if isinstance(this_object, h5py.Group):
                        this_object.visititems(lambda name, obj: sliced.update({name:obj}))
                        sliced.update({'group':group_names[batch]})
                        self.loadObj(sliced)
                        
                    elif isinstance(this_object, h5py.Dataset):
                        sliced.update({group_names[batch]:this_object})

                        self.loadObj(sliced)
        
        except:
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.PopMessageWidgetObj.add_message_stack((('slice error occurred', str(exc_value)), 'error'))

    def addObj(self, data:dict):
        self.loadObj(fullpath=data)
                
    def loadObj(self, fullpath:str, extName=''):
        
        
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
        
        def _getArrowfromLine(v:np.ndarray, color:np.ndarray):
            
            
            if hasattr(color, 'ndim') and color.ndim > 1:
                color = color[..., 1, :]
                color = color.repeat(12, axis=0).reshape(-1, color.shape[-1])
                            
            verctor_line = v[:, 1] - v[:, 0]
            verctor_len = np.linalg.norm(verctor_line, axis=-1, keepdims=True)
            vaild_mask = np.where(verctor_len > 1e-7)
            v = v[vaild_mask[0]]
            verctor_line = v[:, 1] - v[:, 0]
            
            B = len(verctor_line)
            BR = _get_R_between_two_vec(np.array([[0, 0, 1]]).repeat(len(verctor_line), axis=0), verctor_line) # (B, 3, 3)
            temp = Arrow.getTemplate(size=0.01) # (12, 3)
            temp = temp[None, ...].repeat(len(verctor_line), axis=0) # (B, 12, 3)
            BR = BR[:, None, ...].repeat(12, axis=1)
            BR = BR.reshape(-1, 3, 3) # (B*12, 3, 3)
            temp = temp.reshape(-1, 1, 3)
            temp = BR @ temp.transpose(0, 2, 1)
            T = v[:, 1, :][:,None,:]
            T = T.repeat(12, axis=1).reshape(-1, 3)
            vertex = temp.transpose(0, 2, 1).reshape(-1, 3)
            vertex = vertex + T
            
            return Arrow(vertex=vertex, color=color)

        def _rmnan(v):
            v = np.array(v, dtype=np.float32)
            nan_mask = np.logical_not(np.isnan(v))
            rows_with_nan = np.all(nan_mask, axis=1)
            indices = np.where(rows_with_nan)[0]
            return v[indices]
        
        def _getCenter(v:np.ndarray):
            center = v.mean(axis=0)
            if self.center_all is None:
                self.center_all = np.array([center])
            else:
                self.center_all = np.vstack((self.center_all, center))
            # print(self.center_all)

        def _parseArray(k:str, v:np.ndarray):
            
            v = np.nan_to_num(v)
            v = np.float32(v)
            
            
            
            print(k, ':', v.nbytes, 'bytes')
            assert v.nbytes < 1e8, 'array too large, must slice to show'
            
            n_color = self._decode_HexColor_to_RGB(self._isHexColorinName(k))
            user_color = n_color if n_color is not None else self.colormanager.get_next_color()
            
            # -------- lines with arrows
            if (len(v.shape) >= 2 and v.shape[-2] == 2 and v.shape[-1] in (3, 6, 7) and 'line' in k): # (..., 2, 3)
                
                if v.shape[-1] in (6, 7):
                    user_color = v[..., 3:].reshape(-1, 2, v.shape[-1]-3)
                    v = v[..., :3]
            
                
                v = v.reshape(-1, 2, 3)
                lines = Lines(vertex=v, color=user_color)
                
                #-----# Arrow
                if self.ui.checkBox_arrow.isChecked():
                    arrow = _getArrowfromLine(v, user_color)
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
                obj = PointCloud(vertex=v, color=user_color, size=self._isSizeinName(k))
                
                user_color = (user_color,)
                    
            # -------- pointcloud with point-wise color
            elif len(v.shape) >= 2 and v.shape[-1] in (6, 7): # (..., 6)
                vertex = v[..., :3].reshape(-1, 3)
                if v.shape[-1] == 6:
                    color = v[..., 3:6].reshape(-1, 3)
                else:
                    color = v[..., 3:7].reshape(-1, 4)
                    
                obj = PointCloud(vertex=vertex, color=color, size=self._isSizeinName(k))
                
                user_color, per = colorManager.extract_dominant_colors(color, n_colors=3)
                
                
            # -------- coordinate axis
            elif len(v.shape) >= 3 and v.shape[-1] == 4 and v.shape[-2] == 4: # (..., 4, 4)
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
                
            return obj, k, user_color, True

        def _parseDict(k:str, v:dict):
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

            return obj, k, ((0.9, 0.9, 0.9),), False
                        
        def _parseTrimesh(k:str, v:trimesh.parent.Geometry3D):
            
            if isinstance(v, trimesh.Scene):
                print(f'parse Trimesh.Scene, {len(v.geometry)} meshes found')
                _meshlist = []
                for _k, mesh in v.geometry.items():
                    _meshlist.append(mesh)
                v = trimesh.util.concatenate(_meshlist)

                
            
            if isinstance(v, (trimesh.Trimesh)):

                if hasattr(v, 'scale') and v.scale > 100:
                    v.apply_scale(1 / v.scale * 10)
                    self.PopMessageWidgetObj.add_message_stack((('mesh is too large, auto scaled', ''), 'warning'))

                if hasattr(v, 'visual') and hasattr(v.visual, 'material'):
                    if isinstance(v.visual.material, trimesh.visual.material.SimpleMaterial):
                        tex = v.visual.material.image
                    elif isinstance(v.visual.material, trimesh.visual.material.PBRMaterial):
                        tex = v.visual.material.baseColorTexture
                    else:
                        tex = None
                else:
                    tex = None

                vertex_color = v.visual.vertex_colors /255. if isinstance(v.visual, trimesh.visual.color.ColorVisuals) else None
                texcoord = v.visual.uv.view(np.ndarray).astype(np.float32) if isinstance(v.visual, trimesh.visual.texture.TextureVisuals) and hasattr(v.visual, 'uv') and hasattr(v.visual.uv, 'view') else None

                obj = Mesh(v.vertices.view(np.ndarray).astype(np.float32),
                        v.faces.view(np.ndarray).astype(np.int32),
                        norm=v.face_normals.view(np.ndarray).astype(np.float32),
                        color=vertex_color,
                        texture=tex,
                        texcoord=texcoord,
                        faceNorm=True
                        )

                try:
                    if tex is not None and texcoord is not None:
                        texcolor = colorManager.get_color_from_tex(tex, texcoord[:, ::-1])
                        main_colors, per = colorManager.extract_dominant_colors(texcolor, n_colors=3)
                    elif vertex_color is not None:
                        main_colors, per = colorManager.extract_dominant_colors(vertex_color, n_colors=3)
                    else:
                        main_colors = ((0.9, 0.9, 0.9),)
                except:
                    main_colors = ((0.9, 0.9, 0.9),)
                
                return obj, k, main_colors, False



            elif isinstance(v, trimesh.PointCloud):
                if hasattr(v, 'colors') and hasattr(v, 'vertices') and len(v.colors.shape) > 1 and v.colors.shape[0] == v.vertices.shape[0]:
                    if np.max(v.colors) > 1:
                        colors = v.colors / 255.
                    else:
                        colors = v.colors
                    array = np.concatenate((v.vertices, colors), axis=-1)
                    return _parseArray(k, array)
                else:
                    return _parseArray(k, np.array(v.vertices))

                # self.add2ObjPropsTable(v, k, adjustable=True)
                
            else:
                raise ValueError(f'unsupported Trimesh object, {v.__class__.__name__}')

        def _loadNpFile(file):

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

        def _loadFromAny(fullpath, extName):
            if isinstance(fullpath, str) and os.path.isfile(fullpath):
                _extName = os.path.splitext(fullpath)[-1][1:]
                
                if _extName in ['npz', 'npy', 'NPY', 'NPZ',]:
                    obj = _loadNpFile(fullpath)
                elif _extName in ['obj', 'ply', 'stl', 'pcd', 'glb', 'xyz', 'OBJ', 'PLY', 'STL', 'PCD', 'XYZ', 'GLB']:
                    obj = trimesh.load(fullpath, process=False)
                elif _extName in ['h5', 'H5']:
                    obj = h5py.File(fullpath, 'r', track_order=True)
                else:
                    obj = pickle.load(open(fullpath, 'rb'))
                    
                self.setWorkspaceObj(obj)
   
            # load file from API or slice
            elif isinstance(fullpath, (dict)):
                obj = fullpath
   
            # load file from remote 
            elif isinstance(fullpath, (io.BytesIO, io.BufferedReader, io.BufferedWriter)):
                if extName in ['npz', 'npy', 'NPY', 'NPZ',]:
                    obj = _loadNpFile(fullpath)
                else:
                    obj = pickle.load(fullpath)
                
                self.setWorkspaceObj(obj)
                
            else:
                raise ValueError(f'Unknown file type: {type(fullpath)}')
            
            return obj

        
        self.resetObjPropsTable()
        self.colormanager.reset()
        
        try:
            
            # load file from local path
            obj = _loadFromAny(fullpath, extName)
            info = self.formatContentInfo(obj)
            self.ui.label_info.setMarkdown(info)
            self.ui.openGLWidget.reset()
            
            if isinstance(obj, dict):
                
                for k, v in obj.items():
                    k = str(k)
                    if hasattr(v, 'shape'):
                        _v, _k, _c, _isadj = _parseArray(k, v)
                            
                    elif isinstance(v, dict):
                        _v, _k, _c, _isadj = _parseDict(k, v)
                        
                    elif isinstance(v, (trimesh.parent.Geometry3D)):
                        _v, _k, _c, _isadj = _parseTrimesh(k, v)
                        
                    self.ui.openGLWidget.updateObject(ID=_k, obj=_v)
                    self.add2ObjPropsTable(_v, _k, _c, _isadj)
                    

            elif isinstance(obj, (trimesh.parent.Geometry3D)):

                baseName = os.path.basename(fullpath)
                fileName = os.path.splitext(baseName)[0]
                
                _v, _k, _c, _isadj = _parseTrimesh(fileName, obj)
                self.ui.openGLWidget.updateObject(ID=_k, obj=_v)
                self.add2ObjPropsTable(_v, _k, _c, _isadj)

            elif isinstance(obj, h5py.File):

                baseName = os.path.basename(fullpath)
                fileName = os.path.splitext(baseName)[0]
                
                sliced = {}
                group_names = list(obj.keys())
                if len(group_names) > 0:
                    first_object = obj[group_names[0]]
                    if isinstance(first_object, h5py.Group):
                        obj[group_names[0]].visititems(lambda name, obj: sliced.update({name:obj}))
                        sliced.update({'group':group_names[0]})
                        self.loadObj(sliced)
                        
                    elif isinstance(first_object, h5py.Dataset):
                        
                        sliced.update({group_names[0]:first_object})
                        self.loadObj(sliced)

            
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

    def setTrackObject(self, ):
        self.isTrackObject = self.ui.checkBox.isChecked()
        if self.isTrackObject and self.center_all is not None:
            
            self.ui.openGLWidget.camera.translateTo(*self.center_all, isAnimated=True, isEmit=True)

    def backendExeGLCallback(self, func, kwargs):
        getattr(self.ui.openGLWidget, func)(**kwargs)
        
    def backendExeUICallback(self, func, kwargs):
        getattr(self, func)(**kwargs)
    

    def openScript(self, ):
        currentScriptPath = QFileDialog.getOpenFileName(self,"Select Script",self.currentPath, '*.py')[0] # 起始路径

        # print(self.currentScriptPath)

        if len(currentScriptPath) and os.path.isfile(currentScriptPath):
            self.ui.label_script.setText(os.path.basename(currentScriptPath))
            self.currentScriptPath = currentScriptPath
            
    def runScript(self, ):
        self.reset_script_namespace()
        self.ui.openGLWidget.reset()
        self.resetObjPropsTable()
        self.clearObjPropsTable()
        
        if os.path.isfile(self.currentScriptPath):
            fname = os.path.basename(self.currentScriptPath)
            
            sys.path.append(os.path.dirname(self.currentScriptPath))
            with open(self.currentScriptPath, ) as f:

                code = f.read()
                code = code.replace('from pcdviewerAPI import executeSignal', '') # Deprecated
                code = code.replace('import Batch3D', '') # Deprecated

            try:
                exec(code, self.script_namespace)
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
                print(f"--- Error executing script: {fname} ---")
                print(error_details_str)
                print(f"--- End of script error for: {fname} ---")
                
                # 2. 在UI的PopMessageWidget中显示错误摘要
                self.PopMessageWidgetObj.add_message_stack(
                    ((f"Script '{fname}' exec error", str(exc_value)), 'error')
                )
            # self.sendCodeSignal.emit(code, fname)
            # self.ui.pushButton_runscript.disconnect(self)
            # func, kwargs = namespace['main']()
            # # st3 = time.time()
            # # print('t1', st2-st, 't2', st3-st2)
            # getattr(self.ui.openGLWidget, func)(**kwargs)
            # print('namespace:', self.script_namespace)

    def runScriptStateChangeRunning(self, ):
        self.ui.pushButton_runscript.setText('Terminate Script')
        # self.ui.pushButton_runscript.applyStyleSheet(**Button_Style_R)
        self.ui.pushButton_runscript.setIcon(FIF.CANCEL)
        self.ui.pushButton_runscript.disconnect(self)
        self.ui.pushButton_runscript.clicked.connect(self.runScriptTerminate)
        self.ui.widget_circle.start()
        
    def runScriptStateChangeFinish(self, ):
        # print('nb')
        self.ui.pushButton_runscript.setText('Run Script')
        self.ui.pushButton_runscript.setIcon(FIF.SEND)
        # self.ui.pushButton_runscript.applyStyleSheet(**Button_Style_G)
        self.ui.pushButton_runscript.disconnect(self)
        self.ui.pushButton_runscript.clicked.connect(self.runScript)
        self.ui.widget_circle.stop()
        
    def runScriptTerminate(self, ):
        self.quitBackendSignal.emit()
        self.backend.quit()
        self.backend.wait()
        
        self.runScriptStateChangeFinish()
        self.backend.start()

    def getFilePathFromList(self, row:int):
        fullpath = self.ui.tableWidget.item(row, 0).fullpath
        isRemote = self.ui.tableWidget.item(row, 0).isRemote

        return fullpath, isRemote

    def getListLength(self, ):
        return self.ui.tableWidget.rowCount()



    def closeEvent(self, event: QCloseEvent) -> None:
        
        self.saveSettings()
        
        self.backend.quit()
        self.backend.wait(200)
        self.backend.terminate()
        # 
        self.backendSFTPThread.quit()
        self.backendSFTPThread.wait(200)
        self.backendSFTPThread.terminate()
        
        self.reset_script_namespace()
        self.fileDetailUI.close()
        self.remoteUI.close()
        
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
        
        self.ui.tool.setFixedWidth(320)
        self.ui.tool.setFixedHeight(self.height()-50)
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
        print('changeTheme:', theme)
        self.changeTXTTheme(theme)

        setTheme(theme)
        
        self.applyMicaTheme(self.winId())
        
        self.saveSettings()
        
        if not sys.platform == 'win32':
            t = {
                Theme.LIGHT:(0.95,0.95,0.95,1.0),
                Theme.DARK:(0.109, 0.117, 0.125, 1.0),
            }
            self.ui.openGLWidget.setBackgroundColor(t[theme])
            
    def loadSettings(self, ):
        
        try:
            with open(self.configPath, 'r') as f:
                settings = json.load(f)
                
                m = {
                    'Light':Theme.LIGHT,
                    'Dark':Theme.DARK,
                    'Auto':Theme.AUTO
                }
                
                self.tgtTheme = m[settings['theme']]
                
                self.ui.openGLWidget.gl_camera_control_combobox.setCurrentItem(settings['camera'])
                self.ui.openGLWidget.setCameraControl(int(settings['camera']))

                self.currentScriptPath = settings['lastScript']
                if len(self.currentScriptPath) and os.path.isfile(self.currentScriptPath):
                    self.ui.label_script.setText(os.path.basename(self.currentScriptPath))

        except:
            ...

    def saveSettings(self, ):
        
        try:
            settings = {
                'theme':self.tgtTheme.value,
                'camera':self.ui.openGLWidget.gl_camera_control_combobox.currentRouteKey(),
                'lastScript':self.currentScriptPath,
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
            self.openFolder(folderPath)
            self.loadObj(file)
        else:
            self.openFolder(file)
        

def changeGlobalTheme(x):
    global CURRENT_THEME
    CURRENT_THEME = [Theme.LIGHT, Theme.DARK][x]
    


if __name__ == "__main__":
    
    
    
    # setTheme(Theme.AUTO)
    
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

   
    App = App()
    # App.setStyleSheet(style)
    App.setWindowTitle('Batch3D Viewer')
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
            ...

    elif sys.platform == 'darwin':
        font = QFont(['SF Pro Display', 'Helvetica Neue', 'Arial'], 10, QFont.Weight.Normal)
        app.setFont(font)
        
    else:
        font = QFont(['Ubuntu', 'Arial'], 10, QFont.Weight.Normal)
        app.setFont(font)
    
    App.show()

    app.exec()
    
    