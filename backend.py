import pickle
import time, sys
import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, QTime, Signal, Slot, QObject
from PySide6.QtWidgets import QApplication
import traceback
import os
import natsort
import tempfile
import io
import trimesh
import shutil


def remove_left(s:str, sub:str):
    if s.startswith(sub):
        return s[len(sub):]
    return s



def delete_all_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


class backendEngine(QObject):
    executeGLSignal = Signal(str, dict)
    executeUISignal = Signal(str, dict)
    infoSignal = Signal(tuple)
    started = Signal()
    finished = Signal()

    def __init__(self, parent = None) -> None:
        super().__init__(parent)

        self.quitFlag = False
        self.namespace = {}

    def quitLoop(self):
        self.quitFlag = True
        
    def inCodeWapper_print(self, *args, **kwargs):
        if len(args) == 2 and args[0] == 'viewer':
            assert isinstance(args[1], dict), 'print Vertex need a dict'
            
            self.executeGLSignal.emit('updateObject_API', args[1])
        else:
            print(*args, **kwargs)

    def run(self, code:str, fname:str):
        self.started.emit()
        self.namespace = {}
        self.quitFlag = False
        st = time.time()
        ccode = compile(code, fname, 'exec')
        st2 = time.time()
        print('success complied code, cost ', st2-st)

        self.namespace = {'print':self.inCodeWapper_print}
        
        try:
            exec(ccode, self.namespace)
        except:
            traceback.print_exc()
            # exc_type, exc_value, exc_traceback = sys.exc_info()
            self.infoSignal.emit((('脚本异常', traceback.format_exc()), 'error'))
        
        print('success executed code, cost ', time.time()-st2)
        
        if 'loop' in self.namespace.keys():
            while not self.quitFlag:
                QApplication.processEvents()
                try:
                    self.namespace['loop']()
                except:
                    traceback.print_exc()
                    self.infoSignal.emit((('循环异常', traceback.format_exc()), 'error'))
                    self.quitFlag = True
                self.thread().usleep(1)

        del self.namespace
        self.finished.emit()

        

class backendSFTP(QObject):
    executeSignal = Signal(str, dict)
    infoSignal = Signal(tuple)
    listFolderContextSignal = Signal(list, str)

    started = Signal()
    finished = Signal()

    def __init__(self, parent = None) -> None:
        super().__init__(parent)

        self.quitFlag = False
        self.currentDir = '/'
        
        self.localCacheDir = './cache'

    def quitLoop(self):
        self.quitFlag = True
        
    def connectSFTP(self, host, port, username, passwd, dir='/'):
        import paramiko
        self.sftp = paramiko.SSHClient()
        self.sftp.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.sftp.connect(host, port, username, passwd, timeout=5)
        self.sftp = self.sftp.open_sftp()
        self.currentDir = dir
        
        
        self.infoSignal.emit((('服务器连接成功', str('呕吼')), 'complete'))
        # print('connect to sftp server success')
        try:
            self.sftpListDir(dir, False)
        except:
            self.currentDir = '/'
            self.sftpListDir('/', False)
          
        self.executeSignal.emit('serverConnected', {})  
        
        
    def sftpListDir(self, dir, isSet=True, onlydir=False, recursive=False):
        self.currentDir = dir
        files = self.sftp.listdir(dir)
        
        
        # print(self.recurrentListDir(dir))
        
        if onlydir:
            attr = self.sftp.listdir_attr(dir)
            isdir = [a.st_mode & 0o170000 == 0o040000 for a in attr]
            dirs = []
            for i in range(len(files)):
                if isdir[i]:
                    dirs.append(files[i])
            dirs = natsort.natsorted(dirs)
            
        elif recursive:
            files = self.recurrentListDir(dir)
            dirs = natsort.natsorted([remove_left(f, self.currentDir+'/') for f in files])
            
            
        else:
            dirs = natsort.natsorted(files)
            
        # print(files)
        if isSet:
            return 'openRemoteFolder', {'filelist':dirs, 'dirname':self.currentDir}
        else:
            self.listFolderContextSignal.emit(dirs, self.currentDir)
            
            


    def recurrentListDir(self, path):
        files = []
        for name, attr in zip(self.sftp.listdir(path), self.sftp.listdir_attr(path)):
            # full_path = os.path.join(path, name)
            fullpath = self._joinPathLinuxStyle(path, name)
            if attr.st_mode & 0o170000 != 0o040000:
                files.append(fullpath)
            
            else:
                files.extend(self.recurrentListDir(fullpath))
        return files           
           
           
        

    def run(self, callback, kwargs):
        self.started.emit()
        # print(kwargs)
        try:
            rt = getattr(self, callback)(**kwargs)
            if isinstance(rt, (tuple, list)) and len(rt) == 2:
                self.executeSignal.emit(rt[0], rt[1])
        except:
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.infoSignal.emit((('远程错误', str(exc_value)), 'error'))
        # self.quitFlag = False
        finally:
            self.executeSignal.emit('setDownloadProgressHidden', {'hidden':True, })
        
        self.finished.emit()
        
    
    def _downloadProgress(self, a, b):
        self.executeSignal.emit('setDownloadProgress', {'dbytes':a, 'totalbytes':b})
        
    def _exist(self, filename):
        try:
            self.sftp.stat(filename)
            return True
        except:
            return False
        
    @staticmethod
    def _getExtname(filename):
        
        b, e = os.path.splitext(filename)
        return e[1:], b
    
    @staticmethod
    def _joinPathLinuxStyle(*args):
        return '/'.join(args)
        
    def _findAssetFile(self, *filefullpathList):
        
        assetFilePath = []
        # assetFilePathRelative = []
        
        for filefullpath in filefullpathList:
            if not self._exist(filefullpath):
                continue
            #
        
            print('finding Asset file for', filefullpath)
            assetFilePath.append(filefullpath)
            dirName = os.path.dirname(filefullpath)
            baseName = os.path.basename(filefullpath)
            extName, fileName = self._getExtname(baseName)
            
            if extName in ['obj']:
                mtlFileFullPath = self._joinPathLinuxStyle(dirName, fileName+'.mtl')
                if self._exist(mtlFileFullPath):
                    assetFilePath.append(mtlFileFullPath)
                    
                    sftpfile = self.sftp.open(mtlFileFullPath)
                    for line in sftpfile:
                        if line.startswith('map_Kd'):
                            texturePath = line.split()[1]
                            if texturePath.startswith('./'):
                                texturePath = self._joinPathLinuxStyle(dirName, texturePath[2:])
                            else:
                                texturePath = self._joinPathLinuxStyle(dirName, texturePath)
                                
                            assetFilePath.append(texturePath)
                            
                            
        return assetFilePath
            
            
    def _deleteCache(self):
        delete_all_files_in_folder(self.localCacheDir)
            
    def _downLoadtoLoacl(self, remoteName, relativePath):
        tgtPath  = os.path.join(self.localCacheDir, relativePath)
        if not os.path.exists(os.path.dirname(tgtPath)):
            os.makedirs(os.path.dirname(tgtPath))
        
        
        baseName = os.path.basename(remoteName)
        try:
            print('downloading', remoteName, '->', tgtPath)
            self.sftp.get(remoteName, tgtPath, callback=self._downloadProgress, max_concurrent_prefetch_requests=64)
        except:
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.infoSignal.emit((('远程错误', str(exc_value) + '\n' + remoteName + '->' + tgtPath), 'error'))
            
        
    
        
    def downloadFile(self, filename):
        
        self.executeSignal.emit('setDownloadProgressHidden', {'hidden':False, })
        # print('download file', filename)
        # tempfile_path = './temp/temp.pkl'
        # if not os.path.exists('./temp'):
        #     os.mkdir('./temp')
        # if os.path.exists(tempfile_path):
        #     print('rm file start',)
        #     os.remove(tempfile_path)
        #     print('rm file done',)
        # trimesh.load()
        
        remoteName = self.currentDir + '/' + filename
        extName, _ = self._getExtname(filename)
        
        if extName in ['pkl', 'cug', 'npy', 'npz', 'PKL', 'CUG', 'NPY', 'NPZ']:
            # direct cache to memory
            # print(remoteName, '->', 'memory')
            tmpfile = io.BytesIO()
            self.sftp.getfo(remoteName, tmpfile, callback=self._downloadProgress, max_concurrent_prefetch_requests=64)
            tmpfile.seek(0)

            
            return 'loadObj', {'fullpath':tmpfile, 'extName':extName}
        
        elif extName in ['obj']:
            self._deleteCache()
            assetFileList = self._findAssetFile(remoteName)
            
            assetFileListRelative = [remove_left(f, self.currentDir+'/') for f in assetFileList]
            # print('find asset file', assetFileList)
            # print(assetFileListRelative)
            for file, rela in zip(assetFileList, assetFileListRelative):
                self._downLoadtoLoacl(file, rela)
                
            return 'loadObj', {'fullpath':os.path.join(self.localCacheDir, assetFileListRelative[0]), 'extName':extName}
        
    
        elif extName in ['ply', 'txt', 'stl', 'pcd', 'glb', 'xyz', 'PLY', 'TXT', 'STL', 'PCD', 'XYZ', 'GLB']:
            self._deleteCache()
            self._downLoadtoLoacl(remoteName, filename)
            return 'loadObj', {'fullpath':os.path.join(self.localCacheDir, filename), 'extName':extName}
            
                
            