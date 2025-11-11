"""Camera undistortion for sewer pipe inspection."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class CameraIntrinsics:
    """Camera calibration parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    
    @property
    def camera_matrix(self) -> np.ndarray:
        """Intrinsic matrix K."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @property
    def dist_coeffs(self) -> np.ndarray:
        """Distortion coefficients."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)


class IUndistorter(Protocol):
    """Interface for frame undistortion."""
    def undistort(self, frame: np.ndarray) -> np.ndarray: ...


class CameraUndistorter:
    """Removes lens distortion from frames using camera calibration."""
    
    def __init__(self, intrinsics: CameraIntrinsics, alpha: float = 1.0):
        """
        Args:
            intrinsics: Camera calibration parameters
            alpha: Free scaling [0,1]. 0=no invalid pixels, 1=all source pixels
        """
        self._intrinsics = intrinsics
        self._alpha = alpha
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None
        self._frame_shape: Optional[tuple] = None
    
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Undistorted BGR frame
        """
        h, w = frame.shape[:2]
        
        # Initialize undistortion maps on first call or shape change
        if self._map1 is None or self._frame_shape != (h, w):
            self._init_maps(w, h)
        
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
    
    def _init_maps(self, width: int, height: int) -> None:
        """Compute undistortion maps."""
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
            (width, height),
            self._alpha,
            (width, height)
        )
        
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
            None,
            new_camera_matrix,
            (width, height),
            cv2.CV_32FC1
        )
        
        self._frame_shape = (height, width)

