"""
Gerenciador de câmera usando padrão Singleton.
"""
import cv2
import threading
from typing import Optional
from contextlib import contextmanager

from app.logger_config import logger
from app.config import config


class CameraManager:
    """
    Gerenciador singleton para acesso à câmera.
    Garante que apenas uma instância de VideoCapture existe.
    """
    _instance: Optional['CameraManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implementa padrão Singleton thread-safe."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa o gerenciador de câmera."""
        if self._initialized:
            return
        
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._initialized = True
        logger.info("CameraManager inicializado")
    
    def get_camera(self) -> cv2.VideoCapture:
        """
        Obtém ou cria uma instância de VideoCapture.
        
        Returns:
            Instância de cv2.VideoCapture configurada
        """
        if self._video_capture is None or not self._video_capture.isOpened():
            self._video_capture = cv2.VideoCapture(config.camera.device_id)
            self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
            self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
            self._video_capture.set(cv2.CAP_PROP_FPS, config.camera.fps)
            self._video_capture.set(cv2.CAP_PROP_BUFFERSIZE, config.camera.buffer_size)
            logger.info(
                f"Câmera inicializada: {config.camera.width}x{config.camera.height} "
                f"@{config.camera.fps}fps"
            )
        
        return self._video_capture
    
    def release(self) -> None:
        """Libera recursos da câmera."""
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
            logger.info("Recursos da câmera liberados")
    
    @contextmanager
    def camera_context(self):
        """
        Context manager para uso seguro da câmera.
        
        Usage:
            with camera_manager.camera_context() as cap:
                ret, frame = cap.read()
        """
        cap = self.get_camera()
        try:
            yield cap
        finally:
            # Não libera aqui, apenas garante uso seguro
            pass
    
    def __del__(self):
        """Garante liberação de recursos ao destruir."""
        self.release()


# Instância singleton global
camera_manager = CameraManager()

