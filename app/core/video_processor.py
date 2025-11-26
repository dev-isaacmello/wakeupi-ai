"""
Processador de vídeo para detecção de sonolência.
Responsável por processar frames e coordenar detecção, alertas e renderização.
"""
import cv2
import time
import numpy as np
from typing import Generator, Optional

from app.logger_config import logger
from app.config import config
from app.core.camera_manager import camera_manager
from app.detection.eye_detector import EyeDetector
from app.rendering.eye_renderer import EyeRenderer
from app.alert.alert_system import AlertSystem
from app.alert.alert_renderer import AlertRenderer
from app.core.state_manager import StateManager


class VideoProcessor:
    """
    Processador de vídeo para detecção de sonolência.
    
    Coordena detecção de olhos, gerenciamento de estado, alertas e renderização.
    """
    
    def __init__(
        self,
        eye_detector: EyeDetector,
        alert_system: AlertSystem,
        state_manager: StateManager
    ):
        """
        Inicializa o processador de vídeo.
        
        Args:
            eye_detector: Detector de olhos
            alert_system: Sistema de alertas
            state_manager: Gerenciador de estado
        """
        self.detector = eye_detector
        self.alert_system = alert_system
        self.state_manager = state_manager
        
        # Renderizadores
        self.eye_renderer = EyeRenderer(ear_threshold=config.detection.ear_threshold)
        self.alert_renderer = AlertRenderer(alert_system)
        
        logger.info("VideoProcessor inicializado")
    
    def process_frames(self) -> Generator[bytes, None, None]:
        """
        Processa frames de vídeo em loop infinito.
        
        Yields:
            Bytes do frame JPEG codificado para streaming
        """
        cap = camera_manager.get_camera()
        prev_time = time.time()
        
        logger.info("Iniciando processamento de frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Falha ao ler frame da câmera")
                    break
                
                # Espelhar frame para interação natural
                frame = cv2.flip(frame, 1)
                
                # 1. Processamento de detecção
                left_ear, right_ear, landmarks = self.detector.process_frame(frame)
                
                # 2. Lógica de detecção
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
                
                # Determinar estados dos olhos
                # NOTA: Após espelhar o frame, o "Left Eye" do MediaPipe (índices 362)
                # corresponde ao olho DIREITO do usuário (lado direito da imagem).
                # O "Right Eye" do MediaPipe (índices 33) corresponde ao olho ESQUERDO do usuário.
                
                raw_left_closed = False  # MediaPipe Left / User Right
                raw_right_closed = False  # MediaPipe Right / User Left
                
                if left_ear is not None:
                    raw_left_closed = self.detector.is_eye_closed(
                        left_ear,
                        self.detector.left_ear_baseline,
                        frame,
                        landmarks,
                        self.detector.LEFT_EYE_INDICES,
                        self.detector.LEFT_EYE_CONTOUR
                    )
                
                if right_ear is not None:
                    raw_right_closed = self.detector.is_eye_closed(
                        right_ear,
                        self.detector.right_ear_baseline,
                        frame,
                        landmarks,
                        self.detector.RIGHT_EYE_INDICES,
                        self.detector.RIGHT_EYE_CONTOUR
                    )
                
                # Mapear para olhos lógicos do usuário
                user_left_closed = raw_right_closed
                user_right_closed = raw_left_closed
                eyes_closed = user_left_closed and user_right_closed
                
                # Atualizar estado de piscadas
                self.state_manager.update_blink_state(
                    user_left_closed,
                    user_right_closed,
                    current_time
                )
                
                # Lógica de sonolência
                should_alert = self.state_manager.update_drowsiness_state(
                    eyes_closed,
                    current_time
                )
                
                if should_alert:
                    self.alert_system.trigger_alert()
                else:
                    self.alert_system.reset_alert()
                
                self.alert_system.update()
                
                # Calcular taxa de piscadas
                blink_rate = self.state_manager.calculate_blink_rate(current_time)
                
                # Atualizar estatísticas globais
                avg_ear = 0.0
                if left_ear is not None and right_ear is not None:
                    avg_ear = (left_ear + right_ear) / 2
                
                self.state_manager.update_stats(
                    not user_left_closed,
                    not user_right_closed,
                    avg_ear,
                    blink_rate,
                    self.alert_system.alert_active,
                    int(fps)
                )
                
                # 3. Renderização
                if landmarks is not None:
                    self.eye_renderer.draw_debug_circles(
                        frame,
                        landmarks,
                        left_ear,
                        right_ear,
                        self.detector.left_ear_baseline,
                        self.detector.right_ear_baseline
                    )
                
                # Renderizar alertas se ativos
                if self.alert_system.alert_active:
                    self.alert_renderer.render(frame)
                
                # Codificar frame
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logger.warning("Falha ao codificar frame")
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
        
        except Exception as e:
            logger.error(f"Erro no processamento de frames: {e}", exc_info=True)
        finally:
            logger.info("Processamento de frames finalizado")

