"""
Renderizador de alertas visuais.
Responsável apenas pela renderização visual de alertas.
"""
import cv2
import numpy as np
from typing import Optional

from app.alert.alert_system import AlertSystem


class AlertRenderer:
    """
    Renderizador de alertas visuais para frames de vídeo.
    
    Responsabilidade única: Renderizar alertas visuais em frames.
    """
    
    def __init__(self, alert_system: AlertSystem):
        """
        Inicializa o renderizador de alertas.
        
        Args:
            alert_system: Instância do sistema de alertas
        """
        self.alert_system = alert_system
    
    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica overlay visual de alerta no frame.
        
        Args:
            frame: Frame BGR do OpenCV
        
        Returns:
            Frame com overlay de alerta aplicado (modifica in-place)
        """
        if not self.alert_system.alert_active:
            return frame
        
        # Atualizar estado do alerta antes de renderizar
        self.alert_system.update()
        
        h, w = frame.shape[:2]
        flash_state = self.alert_system.flash_state
        
        # Configuração visual baseada no estado do flash
        border_color = (0, 0, 255) if flash_state else (0, 0, 100)
        border_thickness = 30 if flash_state else 15
        
        # Borda pulsante (modificar frame diretamente)
        cv2.rectangle(frame, (0, 0), (w, h), border_color, border_thickness)
        
        # Texto de alerta centralizado
        self._draw_centered_text(frame, "VOCE DORMIU!!!!", 2.0, (0, 0, 255), -50, flash_state)
        self._draw_centered_text(frame, "ACORDE AGORA!!!", 1.2, (255, 255, 255), 50, flash_state)
        
        return frame
    
    def _draw_centered_text(
        self,
        img: np.ndarray,
        text: str,
        scale: float,
        color: tuple,
        y_offset: int,
        flash_state: bool
    ) -> None:
        """
        Desenha texto centralizado no frame.
        
        Args:
            img: Frame BGR
            text: Texto a desenhar
            scale: Escala da fonte
            color: Cor do texto (BGR)
            y_offset: Offset vertical em pixels
            flash_state: Estado atual do flash
        """
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 3
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = (w - text_w) // 2
        y = (h + text_h) // 2 + y_offset
        
        # Fundo do texto para contraste
        bg_color = (0, 0, 0) if flash_state else (20, 20, 20)
        pad = 10
        cv2.rectangle(
            img,
            (x - pad, y - text_h - pad),
            (x + text_w + pad, y + baseline + pad),
            bg_color,
            -1
        )
        
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

