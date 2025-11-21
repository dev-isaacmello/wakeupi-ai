import cv2
import numpy as np
from typing import Tuple, Optional
import time
import threading
import platform
import subprocess

class AlertSystem:
    """
    Sistema de alerta visual e sonoro para sonolência.
    Responsabilidade única: Gerenciar estados de alerta e notificações.
    """
    
    def __init__(self):
        self.alert_active = False
        self.alert_start_time = None
        self.flash_state = False
        self.last_flash_time = 0
        self.flash_interval = 0.2
        self.last_beep_time = 0
        self.beep_interval = 0.5
        
    def trigger_alert(self):
        """Ativa o estado de alerta."""
        if not self.alert_active:
            self.alert_active = True
            self.alert_start_time = time.time()
            # Tocar beep inicial em thread separada
            threading.Thread(target=self._play_beep, daemon=True).start()
            
    def reset_alert(self):
        """Desativa o estado de alerta."""
        self.alert_active = False
        self.alert_start_time = None
        self.flash_state = False
        
    def update(self):
        """Atualiza estados temporais do alerta (flashes, beeps periódicos)."""
        if not self.alert_active:
            return

        current_time = time.time()
        
        # Atualizar flash
        if current_time - self.last_flash_time >= self.flash_interval:
            self.flash_state = not self.flash_state
            self.last_flash_time = current_time
            
        # Tocar beep periódico
        if current_time - self.last_beep_time >= self.beep_interval:
            threading.Thread(target=self._play_beep, daemon=True).start()
            self.last_beep_time = current_time

    def create_alert_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Aplica overlay visual de alerta no frame."""
        if not self.alert_active:
            return frame
            
        # Atualizar lógica de tempo antes de desenhar
        self.update()
        
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Configuração visual baseada no estado do flash
        border_color = (0, 0, 255) if self.flash_state else (0, 0, 100)
        border_thickness = 30 if self.flash_state else 15
        
        # Borda pulsante
        cv2.rectangle(frame_copy, (0, 0), (w, h), border_color, border_thickness)
        
        # Texto de alerta centralizado
        self._draw_centered_text(frame_copy, "VOCE DORMIU!!!!", 2.0, (0, 0, 255), -50)
        self._draw_centered_text(frame_copy, "ACORDE AGORA!!!", 1.2, (255, 255, 255), 50)
        
        return frame_copy

    def _draw_centered_text(self, img, text, scale, color, y_offset):
        """Helper para desenhar texto centralizado."""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 3
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = (w - text_w) // 2
        y = (h + text_h) // 2 + y_offset
        
        # Fundo do texto para contraste
        bg_color = (0, 0, 0) if self.flash_state else (20, 20, 20)
        pad = 10
        cv2.rectangle(img, (x - pad, y - text_h - pad), (x + text_w + pad, y + baseline + pad), bg_color, -1)
        
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def _play_beep(self):
        """Toca som do sistema de forma agnóstica à plataforma."""
        try:
            system = platform.system()
            if system == "Windows":
                import winsound
                winsound.Beep(1000, 200)
            elif system == "Darwin":
                subprocess.Popen(['say', 'beep'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(['beep'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
