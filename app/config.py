"""
Configurações centralizadas do SleepArlet.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraConfig:
    """Configurações da câmera."""
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class DetectionConfig:
    """Configurações de detecção."""
    ear_threshold: float = 0.25
    ear_smoothing_frames: int = 5
    use_deep_learning: bool = False
    deep_learning_check_interval: float = 0.5
    drowsiness_threshold: float = 0.8  # segundos
    blink_debounce: float = 0.15  # segundos


@dataclass
class AlertConfig:
    """Configurações de alerta."""
    flash_interval: float = 0.2  # segundos
    beep_interval: float = 0.5  # segundos
    beep_frequency: int = 1000  # Hz
    beep_duration: int = 200  # ms


@dataclass
class WebConfig:
    """Configurações da aplicação web."""
    host: str = "0.0.0.0"
    port: int = 8000
    websocket_update_interval: float = 0.033  # ~30 FPS


@dataclass
class AppConfig:
    """Configuração principal da aplicação."""
    camera: CameraConfig = None
    detection: DetectionConfig = None
    alert: AlertConfig = None
    web: WebConfig = None
    
    def __post_init__(self):
        """Inicializa configurações padrão se não fornecidas."""
        if self.camera is None:
            self.camera = CameraConfig()
        if self.detection is None:
            self.detection = DetectionConfig()
        if self.alert is None:
            self.alert = AlertConfig()
        if self.web is None:
            self.web = WebConfig()


# Instância global de configuração
config = AppConfig(
    camera=CameraConfig(),
    detection=DetectionConfig(),
    alert=AlertConfig(),
    web=WebConfig()
)

