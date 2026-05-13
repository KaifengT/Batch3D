import math
import copy
from enum import Enum
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer
from .utils.transformations import rotation_matrix, rpy2hRT, invHRT
from .utils.kalman import kalmanFilter
from .utils.transformations import quaternion_from_matrix, quaternion_matrix





class GLCamera(QObject):
    '''
    GLCamera is a class that represents a 3D camera in OpenGL.
    It handles camera trajectory, projection, and view matrices.
    '''
    class controlType(Enum):
        arcball = 0
        trackball = 1
        
    class projectionMode(Enum):
        perspective = 0
        orthographic = 1
        
    
    updateSignal = Signal()
    
    '''
           y
           ^
           |
         __|__
        | -z  |----> x
        |_____|
    
    camera looks to the -z direction
    '''
    def __init__(self) -> None:
        super().__init__()
            
        self.azimuth=135
        self.elevation=-55
        self.viewPortDistance = 10
        self.CameraTransformMat = np.identity(4)
        self.lookatPoint = np.array([0., 0., 0.,])
        self.fx = 1
        self.fy = 1
        self.cx = 0.0
        self.cy = 0.0
        self.intr = np.eye(3)
        
        self.viewAngle = 60.0
        self.near = 0.1
        self.far = 1000.0
        
        self.controltype = self.controlType.trackball
        
        self.archball_rmat = None
        self.target = None
        self.archball_radius = 1.5
        self.reset_flag = False
        
        self.arcboall_quat = np.array([1, 0, 0, 0])
        self.last_arcboall_quat = np.array([1, 0, 0, 0])
        self.arcboall_t = np.array([0, 0, 0])
        
        self.filterAEV = kalmanFilter(3, R=0.4, Q=0.015)
        self.filterlookatPoint = kalmanFilter(3, R=0.4, Q=0.015)
        # BUG TO FIX: filterRotaion cannot deal with quaternion symmetry when R >> Q
        # self.filterRotaion = kalmanFilter(4, Q=0.5, R=0.1)
        self.filterRotaion = kalmanFilter(4, R=0.4, Q=0.015)
        self.filterAngle = kalmanFilter(1)
        
        self.filterPersp = kalmanFilter(16, R=0.5)
        self.filterViewAngle = kalmanFilter(1, R=0.1)
        self.filterNear = kalmanFilter(1, R=0.1)
        self.filterFar = kalmanFilter(1, R=0.1)
        self.filterIntr = kalmanFilter(4, R=0.2, Q=0.02)
        
        
        self.currentProjMatrix = None
        self.targetProjMatrix = None

        self.useCustomIntrinsic = False
        self.customIntrinsicAnimated = True
        self.customIntrinsicCurrent = None
        self.customIntrinsicTarget = None
        self.lastWindowH = 1
        self.lastWindowW = 1
        self.intrinsicPixelOffsetX = 0.0
        self.intrinsicPixelOffsetY = 0.0
        
        self.projection_mode = self.projectionMode.perspective
        
        self.filterAEV.stable(np.array([self.azimuth, self.elevation, self.viewPortDistance]))
        self.filterViewAngle.stable(np.array([self.viewAngle]))
        self.updateTransform(False, False)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateTransform)
        self.timer.setSingleShot(False)
        self.timer.setInterval(7)
        self.timer.start()


        self.timer_proj = QTimer()
        self.timer_proj.timeout.connect(self.updateProjTransform)
        self.timer_proj.setSingleShot(False)
        self.timer_proj.setInterval(7)
        
        self.aspect = 1.0
        
        self.lockRotate = False
        
    def setLockRotate(self, isLock:bool):
        self.lockRotate = isLock

    def setIntrinsicPixelOffset(self, offset_x: float = 0.0, offset_y: float = 0.0):
        self.intrinsicPixelOffsetX = float(offset_x)
        self.intrinsicPixelOffsetY = float(offset_y)
        
    def getIntrinsicPixelOffset(self) -> tuple[float, float]:
        return self.intrinsicPixelOffsetX, self.intrinsicPixelOffsetY
        

    def setCamera(self, azimuth=135, elevation=-55, distance=10, lookatPoint=np.array([0., 0., 0.,])) -> np.ndarray:
        if self.controltype == self.controlType.trackball:
            self.azimuth=azimuth
            self.elevation=elevation
            self.viewPortDistance = distance
            self.lookatPoint = lookatPoint
            
            
        else:
            rmat = rpy2hRT(0, 0, 0, 0, 0, azimuth/180.*math.pi)
            rmat =  rpy2hRT(0, 0, 0, elevation/180.*math.pi, 0, 0) @ invHRT(rmat)
            self.arcboall_quat = quaternion_from_matrix(rmat)

            self.viewPortDistance = distance    
            self.lookatPoint = lookatPoint
            
            
        return self.updateTransform()
    
    @staticmethod
    def _decomposeCameraTransform(transform: np.ndarray):
        mat = np.asarray(transform, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f'Camera transform must be 4x4, got {mat.shape}')

        rmat = mat[:3, :3]
        tvec = mat[:3, 3]
        cam_center = -rmat.T @ tvec

        distance = float(np.linalg.norm(cam_center))
        if distance < 1e-6:
            distance = 1.0

        # camera looks toward -z in camera space, map to world forward direction
        forward = rmat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        lookat = cam_center + forward * distance

        quat = quaternion_from_matrix(mat)
        return distance, lookat.astype(np.float64), quat

    def setCameraTransform(self, transform: np.ndarray, isEmit=True, isAnimated=True) -> np.ndarray:

        mat = np.asarray(transform, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f'Camera transform must be 4x4, got {mat.shape}')

        target_distance, target_lookat, target_quat = self._decomposeCameraTransform(mat)
        current_distance, current_lookat, current_quat = self._decomposeCameraTransform(self.CameraTransformMat)

        if np.dot(target_quat, current_quat) < 0:
            target_quat = -target_quat

        self.controltype = self.controlType.arcball

        self.viewPortDistance = target_distance
        self.lookatPoint = target_lookat
        self.arcboall_quat = target_quat
        self.last_arcboall_quat = current_quat.copy()

        self.filterlookatPoint.stable(current_lookat)
        self.filterAEV.stable(np.array([self.azimuth, self.elevation, current_distance]))
        self.filterRotaion.stable(current_quat)

        if isAnimated:
            if not self.timer.isActive():
                self.timer.start()
        else:
            self.filterlookatPoint.stable(self.lookatPoint)
            self.filterAEV.stable(np.array([self.azimuth, self.elevation, self.viewPortDistance]))
            self.filterRotaion.stable(self.arcboall_quat)
            self.CameraTransformMat = mat.copy()
            if self.timer.isActive():
                self.timer.stop()
        if isEmit:
            self.updateSignal.emit()
        return self.CameraTransformMat

    def _calcFovFromFy(self, fy: float) -> float:
        if fy <= 0:
            raise ValueError('fy must be positive when converting intrinsic to FOV')

        if self.lastWindowH > 1:
            window_h = float(self.lastWindowH)
        elif self.cy > 0:
            window_h = float(self.cy * 2.0)
        else:
            window_h = 1.0

        return 2.0 * math.degrees(math.atan(window_h / (2.0 * float(fy))))

    def _updateCustomIntrinsicState(self, isAnimated=True):
        if not self.useCustomIntrinsic:
            return

        if self.customIntrinsicCurrent is None or self.customIntrinsicTarget is None:
            return

        if isAnimated and self.customIntrinsicAnimated:
            self.customIntrinsicCurrent = self.filterIntr.forward(self.customIntrinsicTarget)
        else:
            self.customIntrinsicCurrent = self.customIntrinsicTarget.copy()
            self.filterIntr.stable(self.customIntrinsicTarget)

        self.fx = float(self.customIntrinsicCurrent[0])
        self.fy = float(self.customIntrinsicCurrent[1])
        self.cx = float(self.customIntrinsicCurrent[2])
        self.cy = float(self.customIntrinsicCurrent[3])

        target_fov = self._calcFovFromFy(self.fy)
        if isAnimated and self.customIntrinsicAnimated:
            self.viewAngle = float(self.filterViewAngle.forward(np.array([target_fov]))[0])
        else:
            self.viewAngle = float(target_fov)
            self.filterViewAngle.stable(np.array([self.viewAngle]))

    def setIntrinsicMatrix(self, K: np.ndarray, isAnimated=True, isEmit=True):
        if self.projection_mode != self.projectionMode.perspective:
            raise ValueError('setIntrinsicMatrix only supports perspective projection mode')

        mat = np.asarray(K, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f'Intrinsic matrix must be 3x3, got shape {mat.shape}')

        if not np.isclose(mat[2, 2], 1.0, atol=1e-6):
            raise ValueError('Intrinsic matrix K[2,2] must be 1.0')

        if abs(float(mat[1, 0])) > 1e-8 or abs(float(mat[2, 0])) > 1e-8 or abs(float(mat[2, 1])) > 1e-8:
            raise ValueError('Intrinsic matrix must use the standard upper-triangular form')

        fx = float(mat[0, 0])
        fy = float(mat[1, 1])
        cx = float(mat[0, 2])
        cy = float(mat[1, 2])

        if not np.isfinite([fx, fy, cx, cy]).all():
            raise ValueError('Intrinsic matrix contains non-finite values')
        if fx <= 0 or fy <= 0:
            raise ValueError('fx and fy must be positive')

        target = np.array([fx, fy, cx, cy], dtype=np.float64)
        self.useCustomIntrinsic = True
        self.customIntrinsicAnimated = bool(isAnimated)
        self.customIntrinsicTarget = target

        if self.customIntrinsicCurrent is None:
            current_cx = self.lastWindowW / 2.0 if self.lastWindowW > 1 else cx
            current_cy = self.lastWindowH / 2.0 if self.lastWindowH > 1 else cy
            self.customIntrinsicCurrent = np.array([self.fx, self.fy, current_cx, current_cy], dtype=np.float64)

        self.filterIntr.stable(self.customIntrinsicTarget)

        if isAnimated:
            if not self.timer_proj.isActive():
                self.timer_proj.start()
        else:
            self._updateCustomIntrinsicState(isAnimated=False)
            self.updateProjTransform(isAnimated=False, isEmit=False)

        if isEmit:
            self.updateSignal.emit()

    def updateIntr(self, window_h, window_w):
        """
        update camera intrinsic matrix based on the window size and view angle
        Args:
            window_h (int): height of the window
            window_w (int): width of the window
        Returns:
            intr (np.ndarray): camera intrinsic matrix
        """

        self.lastWindowH = int(window_h)
        self.lastWindowW = int(window_w)

        self._updateCustomIntrinsicState(isAnimated=self.timer_proj.isActive())

        fov_half_rad = math.radians(self.viewAngle / 2)
        cx = window_w / 2.0
        cy = window_h / 2.0
        # calculate focal length based on the window height and FOV
        
        if self.projection_mode == self.projectionMode.perspective:
            if not self.useCustomIntrinsic:
                self.fy = (window_h / 2.0) / math.tan(fov_half_rad)
                self.fx = self.fy
                self.cx = cx
                self.cy = cy
            
            
        elif self.projection_mode == self.projectionMode.orthographic:
            self.useCustomIntrinsic = False
            ortho_height = self.viewPortDistance * 0.5
            ortho_width = ortho_height * self.aspect
            
            self.fy = window_h / (2.0 * ortho_height)
            self.fx = window_w / (2.0 * ortho_width)
            self.cx = cx
            self.cy = cy
            
        else:
            raise ValueError(f'Unknown projection mode: {self.projection_mode}')



        self.intr = np.array([
            [self.fx, 0,      self.cx + (self.intrinsicPixelOffsetX if self.useCustomIntrinsic else 0.0)],
            [0,       self.fy, self.cy + (self.intrinsicPixelOffsetY if self.useCustomIntrinsic else 0.0)],
            [0,       0,       1.0]
        ], dtype=np.float32)
        
        return self.intr
        
                
        
    def resetAE(self,):
        
        # print(self.elevation, self.azimuth)
        
        if self.elevation > 0:
            counte = self.elevation // 360
            self.elevation -= counte * 360.
        else:
            counte = self.elevation // -360
            self.elevation -= counte * -360.

        if self.azimuth > 0:
            counta = self.azimuth // 360
            self.azimuth -= counta * 360.
        else:
            counta = self.azimuth // -360
            self.azimuth -= counta * -360.
        
    def map2Sphere(self, x, y, height, width):
        cx, cy = width / 2, height / 2  # center of the window
        norm_x = (x - cx) / cx
        norm_y = (cy - y) / cy

        d = math.sqrt(norm_x**2 + norm_y**2)
        if d < self.archball_radius:
            z = math.sqrt(self.archball_radius**2 - d**2)
        else:
            norm = math.sqrt(norm_x**2 + norm_y**2 + 1)
            norm_x /= norm
            norm_y /= norm
            z = 1 / norm

        return np.array([norm_x, norm_y, z])

    def calculateRotation(self, start=[0, 0], end=[0, 0]):
        # params:
        # start: 1x2 norm array, start point of the mouse drag, [x1, y1]
        # end: 1x2 norm array, end point of the mouse drag, [x2, y2]
        axis = np.cross(start, end)
        cos_angle = np.dot(start, end) / (np.linalg.norm(start) * np.linalg.norm(end))
        angle = math.acos(max(min(cos_angle, 1), -1))

        if np.allclose(axis, [0, 0, 0]):
            return [0, 0, 0], 0
        else:
            axis = axis / np.linalg.norm(axis)
            return axis, angle

    def rotationMatrixFromAxisAngle(self, axis, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1 - c
        x, y, z = axis

        return np.array([
            [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
        ])

    def rpyFromRotationMatrix(self, R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.degrees(np.array([x, y, z]))

    def rotate(self, start=0, end=0, window_h=0, window_w=0):
        '''
        Rotate the camera based on mouse drag.
        Args:
            start: 1x2 array, start point of the mouse drag, [x1, y1]
            end: 1x2 array, end point of the mouse drag, [x2, y2]
            window_h: int, height of the window
            window_w: int, width of the window
        Returns: 
            4x4 np.ndarray, rotation matrix
        '''

        # map to sphere
        if self.lockRotate:
            return
        
        if self.controltype == self.controlType.trackball:
            self.azimuth -= float(start) * 0.15
            self.elevation  += float(end) * 0.15

        else:
            start_norm = self.map2Sphere(start[0], start[1], window_h, window_w)
            end_norm = self.map2Sphere(end[0], end[1], window_h, window_w)
            axis, angle = self.calculateRotation(start_norm, end_norm)
            # print('axis, angle', axis, angle)
            # transform screen space to world space
            angle *= 6
            axis = self.CameraTransformMat[:3,:3].T.dot(axis)
            if angle == 0:
                rmat = np.eye(3)
            else:
                rmat = self.rotationMatrixFromAxisAngle(axis, angle)
            # transform 3x3 rmat to 4x4 rmat
            temp_rmat = np.zeros((4,4))
            if rmat.shape == (3, 3, 3):
                print('error rmat shape', rmat.shape)

            temp_rmat[:3,:3] = rmat
            temp_rmat[3,3] = 1
            self.archball_rmat =  temp_rmat
            
            rmat = self.CameraTransformMat[:3,:3] @ rmat.T
            
            # last_quat = quaternion_from_matrix(self.CameraTransformMat)
            
            targetTransformMat = np.identity(4)
            targetTransformMat[:3,:3] = rmat
            tmat = np.identity(4)
            tmat[:3,3] = self.lookatPoint.T
            # print(tmat)
            targetTransformMat = targetTransformMat @ invHRT(tmat)
            self.arcboall_quat = quaternion_from_matrix(targetTransformMat)
            
            # angle = np.dot(last_quat, self.arcboall_quat)
            # if angle < 0:
            #     print('reverse')
            #     self.arcboall_quat = -self.arcboall_quat
            
            self.arcboall_t = targetTransformMat[:3,3]
            
    def zoom(self, ddistance=0):
        self.viewPortDistance -= ddistance * self.viewPortDistance * 0.1
        
    def translate(self, x=0, y=0,):
        
        scale = self.viewPortDistance * 1e-3
        
        xvec = np.array([-scale,0.,0.,0.]).T @ self.CameraTransformMat
        yvec = np.array([0.,scale,0.,0.]).T @ self.CameraTransformMat
        xdelta = xvec * x
        ydelta = yvec * y
        self.lookatPoint += xdelta[:3]
        self.lookatPoint += ydelta[:3]
        
    def translateTo(self, x=0, y=0, z=0, isAnimated=False, isEmit=True):
        self.lookatPoint = np.array([x, y, z,], dtype=np.float32)
        self.updateTransform(isAnimated=isAnimated, isEmit=isEmit)
        
    def rotationMatrix2Quaternion(self, R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        return np.array([qw, qx, qy, qz])

    def quaternion2RotationMatrix(self, q):
        q = q / np.linalg.norm(q)
        qw, qx, qy, qz = q

        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

        return R

    def updateTransform(self, isAnimated=True, isEmit=True) -> np.ndarray:
        # TODO:
        # add timer.stop() when the screen is not moving
        
        if self.controltype == self.controlType.arcball:
    
            if isAnimated:
                
                if not self.timer.isActive():
                    self.timer.start()
                # self.timer.start()
                
                aev_in = np.array([self.azimuth, self.elevation, self.viewPortDistance])
                aev = self.filterAEV.forward(aev_in)
                lookatPoint = self.filterlookatPoint.forward(self.lookatPoint)
                

                if np.dot(self.arcboall_quat, self.last_arcboall_quat) < 0:
                    self.arcboall_quat = -self.arcboall_quat

                quat = self.filterRotaion.forward(self.arcboall_quat)
                if np.allclose(aev, aev_in, atol=1e-3) and \
                    np.allclose(lookatPoint, self.lookatPoint, atol=1e-3) and \
                        np.allclose(self.arcboall_quat, quat, atol=1e-5):

                    aev = aev_in
                    lookatPoint = self.lookatPoint
                    quat = self.arcboall_quat
                    self.filterAEV.stable(aev_in)
                    self.filterlookatPoint.stable(self.lookatPoint)
                    self.filterRotaion.stable(self.arcboall_quat)
                    self.timer.stop()
                                
                

                tmat = np.identity(4)
                tmat[:3,3] = lookatPoint

                self.last_arcboall_quat = copy.deepcopy(self.arcboall_quat)
                self.CameraTransformMat = np.identity(4)
                self.CameraTransformMat = quaternion_matrix(quat)
                
                self.CameraTransformMat[2, 3] = -aev[2]
                
                self.CameraTransformMat = self.CameraTransformMat @ invHRT(tmat)
                

            else:
                
                self.filterlookatPoint.stable(self.lookatPoint)
                
                rmat = rpy2hRT(0,0,0,0,0,self.azimuth/180.*math.pi)
                self.CameraTransformMat =  rpy2hRT(0,0,0,self.elevation/180.*math.pi,0,0) @ np.linalg.inv(rmat) 
                self.CameraTransformMat[2, 3] = -self.viewPortDistance
                
                tmat = np.identity(4)
                tmat[:3,3] = self.lookatPoint.T
                self.CameraTransformMat = self.CameraTransformMat @ invHRT(tmat)
                
            if isEmit:
                self.updateSignal.emit()
            return self.CameraTransformMat            
        else:
            
    
    
            if isAnimated:
                
                if not self.timer.isActive():
                    self.timer.start()
                
                aev_in = np.array([self.azimuth, self.elevation, self.viewPortDistance])
                aev = self.filterAEV.forward(aev_in)
                lookatPoint = self.filterlookatPoint.forward(self.lookatPoint)
                
                if np.allclose(aev, aev_in, atol=1e-3) and np.allclose(lookatPoint, self.lookatPoint, atol=1e-3):
                    # print('stop')
                    self.timer.stop()
                    self.resetAE()
                    self.filterAEV.stable(np.array([self.azimuth, self.elevation, self.viewPortDistance]))
                
                rmat = rpy2hRT(0, 0, 0, 0, 0, aev[0]/180.*math.pi)
                self.CameraTransformMat =  rpy2hRT(0, 0, 0, aev[1]/180.*math.pi, 0, 0) @ invHRT(rmat) 
                self.CameraTransformMat[2, 3] = -aev[2]
                
                tmat = np.identity(4)
                tmat[:3,3] = lookatPoint.T
                self.CameraTransformMat = self.CameraTransformMat @ invHRT(tmat)
                        
            else:
                
                self.filterlookatPoint.stable(self.lookatPoint)
                
                rmat = rpy2hRT(0,0,0,0,0,self.azimuth/180.*math.pi)
                self.CameraTransformMat =  rpy2hRT(0,0,0,self.elevation/180.*math.pi,0,0) @ np.linalg.inv(rmat) 
                self.CameraTransformMat[2, 3] = -self.viewPortDistance
                
                tmat = np.identity(4)
                tmat[:3,3] = self.lookatPoint.T
                self.CameraTransformMat = self.CameraTransformMat @ np.linalg.inv(tmat)
                
            if isEmit:
                self.updateSignal.emit()
            return self.CameraTransformMat
            
    def rayVector(self, ViewPortX=0, ViewPortY=0, dis=1) -> np.ndarray:
        '''
        Calculate the ray vector in world coordinates from screen pixel coordinates
        Args:
            ViewPortX(float): screen pixel X coordinates
            ViewPortY(float): screen pixel Y coordinates
            dis(float): along the ray direction distance
        Returns:
            world_point(np.ndarray(4,)): homogeneous coordinates in world space
        '''
        
        normalized_coords = np.linalg.inv(self.intr) @ np.array([ViewPortX, ViewPortY, 1.0])
        
        if self.projection_mode == self.projectionMode.perspective:
            camera_point = np.array([
                normalized_coords[0] * dis,
                normalized_coords[1] * dis,
                -dis,
                1.0
            ])
        elif self.projection_mode == self.projectionMode.orthographic:
            camera_point = np.array([
                normalized_coords[0],
                normalized_coords[1],
                -dis,
                1.0
            ])
        else:
            raise ValueError(f'Unknown projection mode: {self.projection_mode}')
        
        world_point = np.linalg.inv(self.CameraTransformMat) @ camera_point
        return world_point

    def updateProjTransform(self, isAnimated=True, isEmit=True) -> np.ndarray:
        self._updateCustomIntrinsicState(isAnimated=isAnimated)
        
        if self.projection_mode == self.projectionMode.perspective:
            if self.useCustomIntrinsic and self.lastWindowW > 0 and self.lastWindowH > 0:
                # Use off-axis frustum from intrinsic matrix so principal point shifts
                # are respected in the actual GL projection.
                width = float(self.lastWindowW)
                height = float(self.lastWindowH)
                fx = float(self.fx)
                fy = float(self.fy)
                cx = float(self.cx + self.intrinsicPixelOffsetX)
                cy = float(self.cy + self.intrinsicPixelOffsetY)

                if fx <= 1e-8 or fy <= 1e-8:
                    raise ValueError(f'Invalid focal length for projection: fx={fx}, fy={fy}')

                top = cy * self.near / fy
                bottom = -(height - cy) * self.near / fy
                right = (width - cx) * self.near / fx
                left = -cx * self.near / fx
            else:
                fov_half_rad = np.radians(self.viewAngle / 2)
                top = np.tan(fov_half_rad) * self.near
                bottom = -top
                right = top * self.aspect
                left = -right
            
            rw, rh, rd = 1/(right-left), 1/(top-bottom), 1/(self.far-self.near)
    
            target_matrix = np.array([
                [2 * self.near * rw, 0, 0, 0],
                [0, 2 * self.near * rh, 0, 0],
                [(right+left) * rw, (top+bottom) * rh, -(self.far+self.near) * rd, -1],
                [0, 0, -2 * self.near * self.far * rd, 0]
            ], dtype=np.float32)
            
            
        elif self.projection_mode == self.projectionMode.orthographic:

            height = self.viewPortDistance * 0.5
            width = height * self.aspect
            right = width
            left = -width
            top = height
            bottom = -height
            
            rw, rh, rd = 1/(right-left), 1/(top-bottom), 1/(self.far-self.near)

            target_matrix = np.array([
                [2. * rw, 0, 0, 0],
                [0, 2. * rh, 0, 0],
                [0, 0, -2. * rd, -(self.far+self.near) * rd],
                [0, 0, 0, 1.]
            ], dtype=np.float32).T
            
            
        else:
            raise ValueError(f'Unknown projection mode: {self.projection_mode}')
        
        if isAnimated:
            if self.currentProjMatrix is None:
                self.currentProjMatrix = target_matrix.copy()
                self.filterPersp.stable(target_matrix.flatten())
                return target_matrix
            
            smoothed_matrix = self.filterPersp.forward(target_matrix.flatten())
            self.currentProjMatrix = smoothed_matrix.reshape(4, 4)
            
            if np.allclose(self.currentProjMatrix, target_matrix, atol=2e-5):
                if self.timer_proj.isActive():
                    self.timer_proj.stop()
                    self.filterPersp.stable(target_matrix.flatten())
                    # print('Projection matrix animation stopped.')
                return target_matrix.astype(np.float32)
            
            if not self.timer_proj.isActive():
                self.timer_proj.start()
                # print('Projection matrix animation started.')
                
            if isEmit:
                self.updateSignal.emit()
            
            return self.currentProjMatrix.astype(np.float32)
        else:
            self.currentProjMatrix = target_matrix.copy()
            self.filterPersp.stable(target_matrix.flatten())
            if self.timer_proj.isActive():
                self.timer_proj.stop()
            return target_matrix
        
        
    # def setIntrinsic(self, fx=1.0, fy=1.0, cx=0.0, cy=0.0):
    #     """
    #     Set the intrinsic parameters of the camera.
    #     Args:
    #         fx (float): focal length in x direction
    #         fy (float): focal length in y direction
    #         cx (float): principal point x coordinate
    #         cy (float): principal point y coordinate
    #     """
    #     self.fx = fx
    #     self.fy = fy
    #     self.intr = np.array([[fx, 0, cx],
    #                           [0, fy, cy],
    #                           [0, 0, 1]], dtype=np.float32)
    #     self.updateSignal.emit()
        

    def setFOV(self, fov=60.0):
        self.useCustomIntrinsic = False
        self.viewAngle = fov
        self.filterViewAngle.stable(np.array([self.viewAngle]))
        if not self.timer_proj.isActive():
            self.timer_proj.start()
        self.updateSignal.emit()
        
    def setNear(self, near=0.1):
        if near <= 0.0001:
            near = 0.0001
        self.near = near
        if not self.timer_proj.isActive():
            self.timer_proj.start()
        self.updateSignal.emit()
        
    def setFar(self, far=4000.0):
        if far >= 100000:
            far = 100000
        if far <= self.near + 0.0001:
            far = self.near + 0.0001
        self.far = far
        if not self.timer_proj.isActive():
            self.timer_proj.start()
        self.updateSignal.emit()

    def setAspectRatio(self, aspect_ratio):
        self.aspect = aspect_ratio
        # self.updateSignal.emit()
        
    def setProjectionMode(self, mode):
        if mode not in [self.projectionMode.perspective, self.projectionMode.orthographic]:
            raise ValueError(f'Unknown projection mode: {mode}')
        
        if self.projection_mode != mode:
            self.projection_mode = mode
            if not self.timer_proj.isActive():
                self.timer_proj.start()

    def setViewPreset(self, preset=0):

        presets = {
            0: (90,  -90, self.viewPortDistance), # +X
            1: (-90, -90, self.viewPortDistance), # -X
            2: (180, -90, self.viewPortDistance), # +Y
            3: (0,   -90, self.viewPortDistance), # -Y
            4: (0,     0, self.viewPortDistance), # +Z
            5: (0,   180, self.viewPortDistance), # -Z
        }
        
        
        if preset in presets:
            azimuth, elevation, distance = presets[preset]
            self.setCamera(azimuth=azimuth, elevation=elevation, 
                         distance=distance, lookatPoint=np.array([0., 0., 0.,]))
