
import math
from PySide6.QtCore import ( QPropertyAnimation, QEasingCurve,QTimer,
    QSize, QTime, Qt, Signal, Slot, QRect, QPoint)
from PySide6.QtGui import (QBrush, QColor, QPainter, QPen)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QFormLayout, QLineEdit, QPushButton,
    QWidget, QGraphicsOpacityEffect, QGraphicsDropShadowEffect)
import numpy as np


def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones(
        (1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def rpy2hRT(pos:list):
    """

    :param pos: rpy角转为其次RT矩阵
    :return:
    """
    px = pos[0]
    py = pos[1]
    pz = pos[2]
    roll = pos[3]
    pitch = pos[4]
    yaw = pos[5]

    trans_mat = np.eye(4, 4)

    cyaw = np.cos(yaw)
    syaw = np.sin(yaw)
    cpitch = np.cos(pitch)
    spitch = np.sin(pitch)
    sroll = np.sin(roll)
    croll = np.cos(roll)

    trans_mat[0] = cyaw * cpitch

    trans_mat[0] = [cyaw * cpitch,
                    cyaw * spitch * sroll - syaw * croll,
                    cyaw * spitch * croll + syaw * sroll, px]
    trans_mat[1] = [syaw * cpitch,
                    syaw * spitch * sroll + cyaw * croll,
                    syaw * spitch * croll - cyaw * sroll, py]
    trans_mat[2] = [-spitch, cpitch * sroll, cpitch * croll, pz]
    trans_mat[3] = [0.0, 0.0, 0.0, 1.0]

    return trans_mat



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
     
     
     
class GLAddon_ind(QWidget):

    # send_message_signal = Signal(tuple)
    def __init__(self, parent=None, alpha:int=255):
        super(GLAddon_ind,self).__init__(parent)

        self.setGeometry(QRect(0,0,120,120))


        # self.effect_shadow = QGraphicsDropShadowEffect()
        # self.effect_shadow.setOffset(0,0) # 偏移
        # self.effect_shadow.setBlurRadius(20) # 阴影半径
        # self.effect_shadow.setColor(QColor(60,60,60)) # 阴影颜色
        # self.setGraphicsEffect(self.effect_shadow)

        self.pose = None

        self.pointX = [0.15,0,0]
        self.pointY = [0,0.15,0]
        self.pointZ = [0,0,0.15]

        self._pointX = [-0.15,0,0]
        self._pointY = [0,-0.15,0]
        self._pointZ = [0,0,-0.15]

        self.points = np.array([self.pointX, self.pointY, self.pointZ, self._pointX, self._pointY, self._pointZ]).transpose(1,0)
        # self.points = np.array([self.pointX, self.pointY, self.pointZ]).transpose(1,0)

        self.intrinsics = np.array([300, 0.0, self.width()/2, 0.0, 300, self.height()/2, 0.0, 0.0, 1.0]).reshape(3, 3)

        self.rt = rpy2hRT([0,0,1,0,0,0])

        self.projected_axes = np.array([self.pointX, self.pointY, self.pointZ, self._pointX, self._pointY, self._pointZ])
        # self.projected_axes = np.array([self.pointX, self.pointY, self.pointZ])

        self.transformed_pts = np.array([self.pointX, self.pointY, self.pointZ, self._pointX, self._pointY, self._pointZ]).transpose(1,0)
        # self.transformed_pts = np.array([self.pointX, self.pointY, self.pointZ]).transpose(1,0)

        # self.text_x = QLabel('X', parent= self)
        
        self.pen = QPen()
        self.pen.setWidth(4)
        
        self.pen.setCapStyle(Qt.RoundCap)

        self.blue = (19, 99, 223)
        self.red = (255, 103, 103)
        self.green = (118, 186, 153)

        self._blue = (19, 99, 223, 120)
        self._red = (255, 103, 103, 120)
        self._green = (118, 186, 153, 120)


    @Slot(tuple)
    def set_pose(self, pose):

        # self.rt = rpy2hRT([0,0,0,pose[1] + (math.pi / 2),0,0]) #@ rpy2hRT([0,0,0,-pose[1],0,0])
        # self.rt = self.rt @ rpy2hRT([0,0,0,0,0,-pose[0] - (math.pi / 2)])
        self.rt = pose
        self.rt[2][3] = 1
        # print(self.rt)

        self.transformed_pts = transform_coordinates_3d(self.points, self.rt)
        # print(transformed_pts)

        self.projected_axes = calculate_2d_projections(self.transformed_pts, self.intrinsics)
        
        self.update()
        


    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)


        transformed_pts = self.transformed_pts.transpose().copy()[:,2].tolist()

        index=[]
        sorted_pts = sorted(transformed_pts, reverse=True)
        for i in range(len(transformed_pts)):
            index.append(transformed_pts.index(sorted_pts[i]))

        for i in index:

            per = 1 - (transformed_pts[i] - 0.85) * 1.5
            

            if i == 0:
                
                c = QColor(self.red[0]*per, self.red[1]*per, self.red[2]*per)
                cb =  c
                self.pen.setStyle(Qt.SolidLine)
                # self.text_x.move(self.projected_axes[i][0],self.projected_axes[i][1])
            elif i == 1:
                c = QColor(self.green[0]*per, self.green[1]*per, self.green[2]*per)
                cb =  c
                self.pen.setStyle(Qt.SolidLine)
            elif i == 2:
                c = QColor(self.blue[0]*per, self.blue[1]*per, self.blue[2]*per)
                cb =  c
                self.pen.setStyle(Qt.SolidLine)

            elif i == 3:
                c = QColor(self.red[0]*per, self.red[1]*per, self.red[2]*per, 120)
                cb =  QColor(self.red[0]*per, self.red[1]*per, self.red[2]*per, 80)
                self.pen.setStyle(Qt.DotLine)

            elif i == 4:
                c = QColor(self.green[0]*per, self.green[1]*per, self.green[2]*per, 120)
                cb =  QColor(self.green[0]*per, self.green[1]*per, self.green[2]*per, 80)
                self.pen.setStyle(Qt.DotLine)

            elif i == 5:
                c = QColor(self.blue[0]*per, self.blue[1]*per, self.blue[2]*per, 120)
                cb =  QColor(self.blue[0]*per, self.blue[1]*per, self.blue[2]*per, 80)
                self.pen.setStyle(Qt.DotLine)



            self.pen.setColor(c)
            painter.setBrush(cb)
            painter.setPen(self.pen)
            
            painter.drawLine(self.width() / 2,self.height() / 2,self.projected_axes[i][0],self.projected_axes[i][1])
            if i == 4 or i == 5 or i == 3:
                self.pen.setStyle(Qt.SolidLine)
                painter.setPen(self.pen)
            painter.drawEllipse(self.projected_axes[i][0] - 6, self.projected_axes[i][1] - 6, 12,12)
           
        painter.end()

