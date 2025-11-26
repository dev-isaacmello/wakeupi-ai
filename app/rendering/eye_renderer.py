"""
Renderizador visual para detecção de olhos.
Responsável apenas pela renderização visual de landmarks e estados dos olhos.
"""
import cv2
import numpy as np
import math
import time
from typing import Optional

from app.logger_config import logger


class EyeRenderer:
    """
    Renderizador visual estilo 'Iron Man / Jarvis' para detecção de olhos.
    
    Responsabilidade única: Renderizar visualizações de detecção de olhos.
    """
    
    # Índices MediaPipe para os olhos
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    def __init__(self, ear_threshold: float = 0.25):
        """
        Inicializa o renderizador de olhos.
        
        Args:
            ear_threshold: Threshold EAR para determinar estado do olho
        """
        self.ear_threshold = ear_threshold
        logger.debug("EyeRenderer inicializado")
    
    def draw_debug_circles(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        left_ear: Optional[float],
        right_ear: Optional[float],
        left_baseline: Optional[float] = None,
        right_baseline: Optional[float] = None
    ) -> np.ndarray:
        """
        Desenha visualização estilo HUD avançada nos olhos.
        
        Args:
            frame: Frame BGR do OpenCV
            landmarks: Landmarks do rosto detectados pelo MediaPipe
            left_ear: EAR do olho esquerdo (MediaPipe)
            right_ear: EAR do olho direito (MediaPipe)
            left_baseline: Baseline do olho esquerdo (opcional)
            right_baseline: Baseline do olho direito (opcional)
        
        Returns:
            Frame com visualizações desenhadas (modifica in-place)
        """
        if landmarks is None:
            return frame
        
        # Frame counter simples baseado no tempo do sistema para animação
        t = time.time()
        rotation_angle = (t * 90) % 360  # 90 graus por segundo
        scan_y_offset = (math.sin(t * 3) + 1) / 2  # Oscilação 0 a 1
        
        h_frame, w_frame = frame.shape[:2]
        
        # Overlay transparente para efeitos de brilho
        overlay = frame.copy()
        
        for indices, ear, baseline in [
            (self.LEFT_EYE_INDICES, left_ear, left_baseline),
            (self.RIGHT_EYE_INDICES, right_ear, right_baseline)
        ]:
            if ear is None:
                continue
            
            points = landmarks[indices].astype(int)
            
            # Cálculo de geometria
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            w_eye = x_max - x_min
            h_eye = y_max - y_min
            center = np.mean(points, axis=0).astype(int)
            
            # Estado
            threshold = self.ear_threshold
            if baseline is not None:
                threshold = baseline * 0.65
            is_closed = ear < threshold
            
            # Cores (BGR)
            if is_closed:
                color_primary = (0, 0, 255)      # Red
                color_secondary = (0, 0, 100)    # Dark Red
                color_dim = (0, 0, 50)
            else:
                color_primary = (255, 255, 0)    # Cyan
                color_secondary = (100, 100, 0)  # Dark Cyan
                color_dim = (50, 50, 0)
            
            # 1. Hexágono Reticle (Ao redor da pupila)
            radius = int(max(w_eye, h_eye) * 0.6)
            hex_points = []
            for i in range(6):
                angle_deg = 60 * i + rotation_angle
                angle_rad = math.radians(angle_deg)
                px = int(center[0] + radius * math.cos(angle_rad))
                py = int(center[1] + radius * math.sin(angle_rad))
                hex_points.append([px, py])
            hex_pts = np.array(hex_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [hex_pts], True, color_secondary, 1, cv2.LINE_AA)
            
            # Pontos nos vértices do hexágono
            for pt in hex_points:
                cv2.circle(frame, tuple(pt), 1, color_primary, -1)
            
            # 2. Arcos Dinâmicos (Rotating Arcs)
            axes = (int(radius * 1.4), int(radius * 1.4))
            angle = -rotation_angle * 1.5
            cv2.ellipse(frame, tuple(center), axes, angle, 0, 60, color_primary, 1, cv2.LINE_AA)
            cv2.ellipse(frame, tuple(center), axes, angle, 180, 240, color_primary, 1, cv2.LINE_AA)
            
            # Arco externo 2 (mais fino)
            axes_outer = (int(radius * 1.6), int(radius * 1.6))
            cv2.ellipse(frame, tuple(center), axes_outer, angle * 0.5, 90, 150, color_secondary, 1, cv2.LINE_AA)
            
            # 3. Efeito de Scanline Vertical
            padding = int(w_eye * 0.6)
            x1, y1 = x_min - padding, y_min - padding
            x2, y2 = x_max + padding, y_max + padding
            
            scan_y = int(y1 + (y2 - y1) * scan_y_offset)
            cv2.line(overlay, (x1, scan_y), (x2, scan_y), color_primary, 1)
            
            # 4. Data Callout (Linha de conexão com texto)
            offset_x = 60 if center[0] < w_frame / 2 else -60
            offset_y = -40
            
            callout_end = (center[0] + offset_x, center[1] + offset_y)
            elbow = (center[0] + int(offset_x * 0.3), center[1] + offset_y)
            
            # Linha poligonal
            pts_line = np.array([center, (center[0], center[1] + offset_y), callout_end], np.int32)
            cv2.polylines(frame, [pts_line], False, color_secondary, 1, cv2.LINE_AA)
            
            # Círculo na ponta
            cv2.circle(frame, callout_end, 3, color_primary, -1)
            
            # Texto do EAR no callout
            text_pos = (callout_end[0] + (5 if offset_x > 0 else -80), callout_end[1] - 5)
            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_primary,
                1,
                cv2.LINE_AA
            )
            
            # Status text
            status_text = "CLOSED" if is_closed else "OPEN"
            text_pos_status = (text_pos[0], text_pos[1] + 15)
            cv2.putText(
                frame,
                status_text,
                text_pos_status,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color_secondary,
                1,
                cv2.LINE_AA
            )
        
        # Aplicar overlay com transparência (efeito glow/scan)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame

