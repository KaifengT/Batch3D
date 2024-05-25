
import sys, os
wdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(wdir)
import numpy as np
import numpy.linalg as linalg
from PySide6.QtWidgets import ( QApplication, QMainWindow, QTableWidgetItem, QWidget, QFileDialog, QDialog)
from PySide6.QtCore import  QThread, Signal, Qt
from PySide6.QtGui import QCloseEvent, QIcon, QFont, QAction
from ui.PopMessageWidget import PopMessageWidget_fluent as PopMessageWidget
import multiprocessing


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

try:
    from win32mica import ApplyMica, MicaTheme, MicaStyle
except:
    ...

########################################################################



from qfluentwidgets import (setTheme, Theme, setThemeColor, qconfig, RoundMenu)
from qfluentwidgets import FluentIcon as FIF


class cellWidget(QTableWidgetItem):
    def __init__(self, text: str, fullpath='', isRemote=False) -> None:
        
        self.fullpath = fullpath
        self.isRemote = isRemote
        return super().__init__(text,)

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
                             
        self.ui.pushButton_connect.clicked.connect(self.connectSFTP)
        self.ui.pushButton_openfolder.clicked.connect(self.openFolder)
        
        self.ui.pushButton_cancel.clicked.connect(self.close)
        
        # self.ui.pushButton_connect.applyStyleSheet(**Button_Style_GS)
        # self.ui.pushButton_openfolder.applyStyleSheet(**Button_Style_GS)
        # self.ui.pushButton_cancel.applyStyleSheet(**Button_Style_R)
        
        self.ui.tableWidget.cellDoubleClicked.connect(self.chdirSFTP)
        self.ui.tableWidget.setHorizontalHeaderLabels(['目录'])
        
        self.ui.pushButton_openfolder.setDisabled(True)
        
        self.configPath = './ssh.config'
        
    def serverConnected(self, ):
        self.ui.pushButton_openfolder.setDisabled(False)
        
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
        fullpath = self.ui.tableWidget.item(row, col).fullpath
        
        self.executeSignal.emit('sftpListDir', {'dir':fullpath, 'isSet':False, 'onlydir':True})
        
    def openFolder(self, ):
        
        self.executeSignal.emit('sftpListDir', {'dir':self.ui.lineEdit_dir.text(), 'recursive':(False, True)[self.ui.comboBox.currentIndex()]})
        self.close()
        
    def setFolderContents(self, filelist:list, dirname:str):
        self.ui.tableWidget.setRowCount(0)
        # self.ui.tableWidget.setHorizontalHeaderLabels([dirname])
        self.ui.lineEdit_dir.setText(dirname)
        for f in filelist:
            self.ui.tableWidget.insertRow(0)
            
            event_widget = cellWidget(f, dirname.rstrip('/') + '/' + f, True)
            self.ui.tableWidget.setItem(0, 0, event_widget)
            
        self.ui.tableWidget.insertRow(0)
        event_widget = cellWidget('..', os.path.dirname(dirname), True)
        self.ui.tableWidget.setItem(0, 0, event_widget)
            
        

            
        
    def closeEvent(self, event: QCloseEvent) -> None:
        self.closedSignal.emit()
        self.saveSettings()
        return super().closeEvent(event)
    
    def showEvent(self, event: QCloseEvent) -> None:
        self.showSiganl.emit()
        
        self.loadSettings()
        return super().showEvent(event)
        

class App(QMainWindow):
    # ----------------------------------------------------------------------
    sendCodeSignal = Signal(str, str)
    sftpSignal = Signal(str, dict)
    quitBackendSignal = Signal()

    def __init__(self,):
        """"""
        super().__init__()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.PopMessageWidgetObj = PopMessageWidget(self)
        self.windowBlocker = windowBlocker(self) 
        
        
        self.resize(1600,900)
        
        self.currentPath = './'
        
        self.isNolyBw = True
        
        self.currentScriptPath = ''

        self.ui.pushButton_openfolder.clicked.connect(self.openFolder)
        self.ui.pushButton_openfolder.setIcon(FIF.FOLDER)
        self.ui.pushButton_openremotefolder.setIcon(FIF.CLOUD_DOWNLOAD)
        self.ui.tableWidget.currentCellChanged.connect(self.cellClickedCallback)
        self.ui.tableWidget.cellDoubleClicked.connect(self.cellClickedCallback)
        # self.ui.checkBox.stateChanged.connect(self.switchExtCallback)
        # self.ui.checkBox_2.stateChanged.connect(self.switchOblyBwCallback)
        self.ui.doubleSpinBox.valueChanged.connect(self.setGLScale)

        self.ui.pushButton_openscript.clicked.connect(self.openScript)
        self.ui.pushButton_openscript.setIcon(FIF.CODE)
        self.ui.pushButton_runscript.clicked.connect(self.runScript)
        
        # self.ui.pushButton_openremotefolder.applyStyleSheet(**Button_Style_GS)


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


        self.backendEngine.executeGLSignal.connect(self.backendExeGLCallback)
        self.backendEngine.executeUISignal.connect(self.backendExeUICallback)
        self.backendEngine.infoSignal.connect(self.PopMessageWidgetObj.add_message_stack)
        self.sendCodeSignal.connect(self.backendEngine.run)
        

        # self.ui.pushButton_runscript.applyStyleSheet(**Button_Style_G)

        self.backendEngine.started.connect(self.runScriptStateChangeRunning)
        self.backendEngine.finished.connect(self.runScriptStateChangeFinish)
        self.quitBackendSignal.connect(self.backendEngine.quitLoop)

        self.backend.start()
        self.backendSFTPThread.start()

        # self.switchExtCallback()

        self.isTrackObject = True
        self.ui.checkBox.setChecked(self.isTrackObject)
        self.ui.checkBox.setOnText('开')
        self.ui.checkBox.setOffText('关')
        self.ui.checkBox.checkedChanged.connect(self.setTrackObject)
        self.ui.checkBox_axis.checkedChanged.connect(self.setGLAxisVisable)
        self.ui.checkBox_axis.setOnText('开')
        self.ui.checkBox_axis.setOffText('关')
        self.ui.checkBox_arrow.setOnText('开')
        self.ui.checkBox_arrow.setOffText('关')

        self.center_all = None
        
        
        self.ui.pushButton_openremotefolder.clicked.connect(self.openRemoteUI)
        
        self.remoteUI.closedSignal.connect(lambda:self.windowBlocker.setHidden(True))
        self.remoteUI.showSiganl.connect(lambda:self.windowBlocker.setHidden(False))
        self.backendSFTP.listFolderContextSignal.connect(self.remoteUI.setFolderContents)
        
        self.ui.pushButton_runscript.setIcon(FIF.SEND)
        
        
        
        
        
        
        
        self.themeToolButtonMenu = RoundMenu(parent=self)
        self.themeLIGHTAction = QAction(FIF.BRIGHTNESS.icon(), '浅色')
        self.themeDARKAction = QAction(FIF.QUIET_HOURS.icon(), '深色')
        self.themeAUTOAction = QAction(FIF.CONSTRACT.icon(), '自动')
        self.themeToolButtonMenu.addAction(self.themeLIGHTAction)
        self.themeToolButtonMenu.addAction(self.themeDARKAction)
        self.themeToolButtonMenu.addAction(self.themeAUTOAction)
        self.ui.toolButton_theme.setMenu(self.themeToolButtonMenu)
        self.ui.toolButton_theme.setIcon(FIF.PALETTE)
        self.tgtTheme = Theme.AUTO
        
        self.themeLIGHTAction.triggered.connect(lambda:self.changeTheme(Theme.LIGHT))
        self.themeDARKAction.triggered.connect(lambda:self.changeTheme(Theme.DARK))
        self.themeAUTOAction.triggered.connect(lambda:self.changeTheme(Theme.AUTO))
        
        
        self.configPath = './user.config'
        self.loadSettings()
        # self.changeTheme(self.tgtTheme)
        self.changeTXTTheme(self.tgtTheme)
        
        self.setUpWatchForThemeChange()
        
    def openFolder(self, ):
        self.currentPath = QFileDialog.getExistingDirectory(self,"选取文件夹",self.currentPath) # 起始路径
        
        # print(self.currentPath)
        if len(self.currentPath) and os.path.exists(self.currentPath):
            filelist = os.listdir(self.currentPath)
            filelist = natsort.natsorted(filelist, )
            filelist = filelist[::-1]
            self.ui.tableWidget.setRowCount(0)
            for f in filelist:
                
                self.addFiletoTable(f)
                
    def openRemoteFolder(self, filelist:list, dirname:str):
        # print(filelist)
        
        self.ui.tableWidget.setRowCount(0)
        filelist = filelist[::-1]
        for f in filelist:
            self.addFiletoTable(f, isRemote=True)


        
    def addFiletoTable(self, filepath:str, isRemote=False):
        extName = os.path.splitext(filepath)[-1][1:]
        
        if extName in ['obj', 'pkl', 'cug', 'npy', 'npz', 'OBJ', 'PKL', 'CUG', 'NPY', 'NPZ', 'ply', 'stl', 'pcd', 'glb', 'xyz', 'PLY', 'STL', 'PCD', 'XYZ', 'GLB'] and not filepath.startswith('.'):
        
            self.ui.tableWidget.insertRow(0)
            event_widget = cellWidget(filepath, os.path.join(self.currentPath, filepath), isRemote=isRemote)
            self.ui.tableWidget.setItem(0, 0, event_widget)


    def formatContentInfo(self, obj):
        textInfo = ' FILE CONTENT: '.center(50, '-') + '\n\n'
        
        def fill_tab_before_return(text, tab=4):
            ...
            
        try:
            if isinstance(obj, dict):
                
                for k, v in obj.items():
                    if isinstance(v, np.ndarray):
                        textInfo += f'{k} : ndarray{v.shape}\n\n'
                        textInfo += str(v) + '\n'
                        
                    elif isinstance(v, (dict)):
                        textInfo += f'{k} : {v.__class__}\n\n'
                        for kk, vv in v.items():
                            textInfo += f'\t|-{kk} : {vv.__class__}\n'
                    
                    else:
                        textInfo += f'{k} : {v.__class__}\n\n'
                        textInfo += str(v) + '\n'
                        
                        
                    textInfo += '\n' + '-' * 50 + '\n\n'
                
            else:
                textInfo += 'unsupport object'
        except:
            textInfo += 'ERROR'
        finally:
            return textInfo

    def _isHexColorinName(self, name) -> str:
        if '#' in name:
            name = name.split('#', 2)[1]
            name = name.split('_', 2)[0]
            if len(name) == 6 or len(name) == 8:
                return name
            else:
                return 'FFFFFF'
        else:
            return 'FFFFFF'
        
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
        
    def loadObj(self, fullpath:str, extName=''):
        
        
        def get_R_between_two_vec(src, dst):
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

        def _dealArray(v:np.ndarray):
            v = _rmnan(v)
            v = np.float32(v)
            
            # -------- pointcloud
            if len(v.shape) == 2 and v.shape[1] == 3: # (N, 3)
                
                obj = PointCloud(vertex=v, color=self._isHexColorinName(k), size=self._isSizeinName(k))
                self.ui.openGLWidget.updateObject(ID=k, obj=obj)
                # self.ui.openGLWidget.updateObject(ID=k, vertex=v, color=self._isHexColorinName(k), size=self._isSizeinName(k), type='pointcloud')
                _getCenter(v)
                    
            # -------- pointcloud with point-wise color
            elif len(v.shape) == 2 and v.shape[1] in [6, 7]: # (N, 6)
                vertex = v[:, :3]
                color = v[:, 3:]
                obj = PointCloud(vertex=vertex, color=color, size=self._isSizeinName(k))
                self.ui.openGLWidget.updateObject(ID=k, obj=obj)
                # self.ui.openGLWidget.updateObject(ID=k, vertex=vertex, color=color, size=self._isSizeinName(k), type='pointcloud')
                
                _getCenter(vertex)
            
            # --------- lines with arrows
            elif len(v.shape) == 3 and v.shape[1] == 2 and v.shape[2] == 3: # (N, 2, 3)
                
                lines = Lines(vertex=v, color=self._isHexColorinName(k))
                
                #-----# Arrow
                if self.ui.checkBox_arrow.isChecked():
                    verctor_line = v[:, 1] - v[:, 0]
                    verctor_len = np.linalg.norm(verctor_line, axis=-1, keepdims=True)
                    vaild_mask = np.where(verctor_len > 1e-7)
                    v = v[vaild_mask[0]]
                    verctor_line = v[:, 1] - v[:, 0]
                    
                    B = len(verctor_line)
                    BR = get_R_between_two_vec(np.array([[0, 0, 1]]).repeat(len(verctor_line), axis=0), verctor_line) # (B, 3, 3)
                    temp = Arrow.getTemplate() # (12, 3)
                    
                    temp = temp[None, ...].repeat(len(verctor_line), axis=0) # (B, 12, 3)
                    
                    
                    BR = BR[:, None, ...].repeat(12, axis=1)
                    
                    BR = BR.reshape(-1, 3, 3) # (B*12, 3, 3)
                    
                    temp = temp.reshape(-1, 1, 3)
                    
                    temp = BR @ temp.transpose(0, 2, 1)
                    T = v[:, 1, :][:,None,:]
                    T = T.repeat(12, axis=1).reshape(-1, 3)
                    
                    vertex = temp.transpose(0, 2, 1).reshape(-1, 3)
                    vertex = vertex + T
                    
                    arrow = Arrow(vertex=vertex, color=self._isHexColorinName(k))
                    
                    obj = UnionObject()
                    obj.add(arrow)
                    obj.add(lines)
                    
                    self.ui.openGLWidget.updateObject(ID=k, obj=obj)
                else:
                    self.ui.openGLWidget.updateObject(ID=k, obj=lines)
                
                
                
                
        
            elif len(v.shape) == 3 and v.shape[1] == 8 and v.shape[2] == 3: # (N, 8, 3)
                obj = BoundingBox(vertex=v, color=self._isHexColorinName(k))
                self.ui.openGLWidget.updateObject(ID=k, obj=obj)
                # self.ui.openGLWidget.updateObject(ID=k, vertex=v, color=self._isHexColorinName(k), type='boundingbox')

        def _dealDict(v:dict):
            assert 'vertex' in v.keys(), '网格缺少顶点(vertex)'
            assert 'face' in v.keys(), '网格缺少面片索引(face)'
            
            obj = Mesh(v['vertex'], v['face'])
            self.ui.openGLWidget.updateObject(ID=k, obj=obj)
            


        try:
            # with open(fullpath, 'rb') as f:
            if isinstance(fullpath, str) and os.path.isfile(fullpath):
                extName = os.path.splitext(fullpath)[-1][1:]
                # if extName == '...':
                    # f = PLYLoader(fullpath)
                    # obj = f.data
                if extName in ['npz', 'npy', 'NPY', 'NPZ',]:
                    obj = np.load(fullpath)
                    obj = dict(obj)
                elif extName in ['obj', 'ply', 'stl', 'pcd', 'glb', 'xyz', 'OBJ', 'PLY', 'STL', 'PCD', 'XYZ', 'GLB']:
                    obj = trimesh.load(fullpath)
                else:
                    obj = pickle.load(open(fullpath, 'rb'))
   
            else:
                if extName in ['npz', 'npy', 'NPY', 'NPZ',]:
                    obj = np.load(fullpath)
                    obj = dict(obj)
                else:
                    obj = pickle.load(fullpath)
                
            
            info = self.formatContentInfo(obj)
            self.ui.label_info.setText(info)
            # self.ui.label_info.setMarkdown(info)
            
            if isinstance(obj, dict):
                
                self.center_all = None
                
                self.ui.openGLWidget.reset()
                for k, v in obj.items():
                    if hasattr(v, 'shape'):
                        # if len(v.shape) in [3, 4]:
                        #     for batch in v:
                        #         _dealArray(batch)
                        if len(v.shape) in [3, 2]:
                            _dealArray(v)
                            
                    elif isinstance(v, dict):
                        _dealDict(v)

                    
                if self.isTrackObject and self.center_all is not None:
                    self.center_all = self.center_all.mean(axis=0) * self.ui.openGLWidget.scale
                    # print(self.center_all)
                    self.ui.openGLWidget.camera.translateTo(*self.center_all, )

            elif isinstance(obj, (trimesh.parent.Geometry3D)):
                self.ui.openGLWidget.reset()
                baseName = os.path.basename(fullpath)
                fileName = os.path.splitext(baseName)[0]
                self.ui.openGLWidget.updateTrimeshObject(ID=fileName, obj=obj)

                
            
        except:
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.PopMessageWidgetObj.add_message_stack((('文件加载错误', str(exc_value)), 'error'))
        
    def switchExtCallback(self, checked=True):
        if checked:
            self.ui.openGLWidget.baseTransform = np.array([ [-1.31165008e-01,  1.64332643e-02,  9.91064088e-01,  2.52000000e+00],
                                                            [ 1.36828413e-02,  9.99839251e-01, -1.48635603e-02,  9.36000000e+00],
                                                            [-9.91127227e-01,  5.41885955e-03, -1.31439524e-01,  1.78900000e+01],
                                                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
                        ])
            
        else:
            self.ui.openGLWidget.baseTransform = np.identity(4, dtype=np.float32)
            
        self.ui.openGLWidget.update()
        
    def setGLScale(self,):
        scale = self.ui.doubleSpinBox.value()
        self.ui.openGLWidget.scale = scale
        self.ui.openGLWidget.update()
        
    def setTrackObject(self, ):
        self.isTrackObject = self.ui.checkBox.isChecked()
        if self.isTrackObject and self.center_all is not None:
            
            self.ui.openGLWidget.camera.translateTo(*self.center_all, isAnimated=True, isEmit=True)

    def setGLAxisVisable(self, ):
        self.ui.openGLWidget.isAxisVisable = self.ui.checkBox_axis.isChecked()
        self.ui.openGLWidget.update()
        
    def switchOblyBwCallback(self, ):
        self.isNolyBw = self.ui.checkBox_2.isChecked()

    # def updateBoundingBox(self, ID=1, vertex:np.ndarray=None):
    #     try:
    #         self.ui.openGLWidget.updateboundingbox(ID, vertex)

    #     except:
    #         exc_type, exc_value, exc_traceback = sys.exc_info()
    #         self.PopMessageWidgetObj.add_message_stack((('数据错误', str(exc_value)), 'error'))

    # def updatePointCloud(self, ID=1, pcd=None):
    #     try:
    #         self.ui.openGLWidget.updatepointcloud(ID, pcd)

    #     except:
    #         exc_type, exc_value, exc_traceback = sys.exc_info()
    #         self.PopMessageWidgetObj.add_message_stack((('数据错误', str(exc_value)), 'error'))


    def backendExeGLCallback(self, func, kwargs):
        getattr(self.ui.openGLWidget, func)(**kwargs)
        
    def backendExeUICallback(self, func, kwargs):
        getattr(self, func)(**kwargs)
    

    def openScript(self, ):
        currentScriptPath = QFileDialog.getOpenFileName(self,"选取脚本",self.currentPath, '*.py')[0] # 起始路径

        # print(self.currentScriptPath)

        if len(currentScriptPath) and os.path.isfile(currentScriptPath):
            self.ui.label_script.setText(os.path.basename(currentScriptPath))
            self.currentScriptPath = currentScriptPath
            
    def runScript(self, ):
        if os.path.isfile(self.currentScriptPath):
            fname = os.path.basename(self.currentScriptPath)
            namespace = {}
            sys.path.append(os.path.dirname(self.currentScriptPath))
            with open(self.currentScriptPath, ) as f:

                code = f.read()
                code = code.replace('from pcdviewerAPI import executeSignal', '')
                
                self.sendCodeSignal.emit(code, fname)
                self.ui.pushButton_runscript.disconnect(self)
                # func, kwargs = namespace['main']()
                # # st3 = time.time()
                # # print('t1', st2-st, 't2', st3-st2)
                # getattr(self.ui.openGLWidget, func)(**kwargs)

    def runScriptStateChangeRunning(self, ):
        # self.ui.pushButton_runscript.setText('终止')
        # self.ui.pushButton_runscript.applyStyleSheet(**Button_Style_R)
        self.ui.pushButton_runscript.setIcon(FIF.CANCEL)
        self.ui.pushButton_runscript.disconnect(self)
        self.ui.pushButton_runscript.clicked.connect(self.runScriptTerminate)
        self.ui.widget_circle.start()
        
    def runScriptStateChangeFinish(self, ):
        # print('nb')
        # self.ui.pushButton_runscript.setText('运行')
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
        
    def closeEvent(self, event: QCloseEvent) -> None:
        
        self.saveSettings()
        
        self.backend.quit()
        self.backend.wait(200)
        self.backend.terminate()
        # 
        self.backendSFTPThread.quit()
        self.backendSFTPThread.wait(200)
        self.backendSFTPThread.terminate()
        
        return super().closeEvent(event)
    

    def openRemoteUI(self, ):
        if sys.platform == 'win32':
            try:
                m = {
                    Theme.LIGHT:MicaTheme.LIGHT,
                    Theme.DARK:MicaTheme.DARK,
                    Theme.AUTO:MicaTheme.AUTO
                }

                ApplyMica(self.remoteUI.winId(), m[self.tgtTheme], MicaStyle.DEFAULT)
            except:
                ...
        self.remoteUI.show()
        
    def setDownloadProgress(self, dbytes:int, totalbytes:int, isBytes=True):
        self.ui.openGLWidget.statusbar.setProgress(dbytes, totalbytes, isBytes)
        
    def setDownloadProgressHidden(self, hidden:bool):
        self.ui.openGLWidget.statusbar.setHidden(hidden)
        
    def resizeEvent(self, event: QCloseEvent) -> None:
        self.windowBlocker.resize(self.size())        
        return super().resizeEvent(event)

    def serverConnected(self, ):
        self.remoteUI.serverConnected()
        
    def setUpWatchForThemeChange(self, ):
        self.watchForThemeTimer = QTimer()
        self.watchForThemeTimer.timeout.connect(self.watchForThemeChange)
        self.watchForThemeTimer.start(500)
        self.watchForThemeTimer.setSingleShot(False)
        # self.currentTheme = qconfig.theme
    
    def watchForThemeChange(self, ):
        global CURRENT_THEME
        if CURRENT_THEME != qconfig.theme and self.tgtTheme == Theme.AUTO:
            
            setTheme(CURRENT_THEME)
            
    def changeTXTTheme(self, theme):
        global CURRENT_THEME
        if theme == Theme.LIGHT:
            label_info_color = '#202020'
        elif theme == Theme.DARK:
            label_info_color = '#E0E0E0'
        else:
            if CURRENT_THEME == Theme.LIGHT:
                label_info_color = '#202020'
            else:
                label_info_color = '#E0E0E0'
        
        
        self.ui.label_info.setStyleSheet(
            '''
            QTextBrowser
                {{
                    background-color: #05DDDDDD;
                    
                    border-radius: 6px;
                    border: 0px;
                    font: 1000 8pt;
                    
                    color: {0};
                }}
            '''.format(label_info_color)
        )
            
    def changeTheme(self, theme):
        global CURRENT_THEME
        self.tgtTheme = theme
        
        self.changeTXTTheme(theme)
        
        # print('changeTheme', 'CURRENT_THEME', CURRENT_THEME, 'qconfig.theme', qconfig.theme, 'tgtTheme', self.tgtTheme, 'theme', theme,)
        
        if theme == Theme.AUTO:
            setTheme(theme)
            CURRENT_THEME = qconfig.theme
            self.watchForThemeTimer.start(1000)
            self.watchForThemeTimer.setSingleShot(False)
        else:
            setTheme(theme)
            self.watchForThemeTimer.stop()
                
        # setTheme(theme)
        
        if sys.platform == 'win32':
            try:
                m = {
                    Theme.LIGHT:MicaTheme.LIGHT,
                    Theme.DARK:MicaTheme.DARK,
                    Theme.AUTO:MicaTheme.AUTO
                }
                if theme == Theme.AUTO:
                    ApplyMica(self.winId(), m[theme], MicaStyle.DEFAULT, changeGlobalTheme)
                else:
                    ApplyMica(self.winId(), m[theme], MicaStyle.DEFAULT)

            except:
                ...
        
        self.saveSettings()
        
            
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
                
        except:
            ...

    def saveSettings(self, ):
        
        try:
            settings = {
                'theme':self.tgtTheme.value,
            }
            
            with open(self.configPath, 'w') as f:
                
                json.dump(settings, f, indent=4)
        except:
            ...

style = """
QMainWindow {
    background-color: #353535;

    border-top-left-radius: 2px;
    border-top-right-radius: 2px;
    border-bottom-left-radius: 2px;
    border-bottom-right-radius: 2px;

}
"""    


def changeGlobalTheme(x):
    global CURRENT_THEME
    CURRENT_THEME = [Theme.LIGHT, Theme.DARK][x]
    


if __name__ == "__main__":
    
    
    
    setTheme(Theme.AUTO)
    
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

    font = QFont([u'Cascadia Mono', u'Microsoft Yahei UI'], )
    app.setFont(font)
   
    App = App()
    # App.setStyleSheet(style)
    App.setWindowTitle('Batch3D Viewer')
    App.setWindowIcon(QIcon('icon.ico'))

    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
        try:
            # from win32mica import ApplyMica, MicaTheme, MicaStyle
            App.setAttribute(Qt.WA_TranslucentBackground)
            ApplyMica(App.winId(), MicaTheme.AUTO, MicaStyle.DEFAULT, changeGlobalTheme)

        except:
            ...

    elif sys.platform == 'darwin':
        pass
    
    # setTheme(Theme.LIGHT)
    
    App.show()

    app.exec()
    
    