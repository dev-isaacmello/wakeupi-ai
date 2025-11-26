"""
Módulos core da aplicação.
Gerencia câmera, estado e processamento de vídeo.
"""

from .camera_manager import CameraManager, camera_manager
from .state_manager import StateManager
from .video_processor import VideoProcessor

__all__ = [
    'CameraManager',
    'camera_manager',
    'StateManager',
    'VideoProcessor'
]

