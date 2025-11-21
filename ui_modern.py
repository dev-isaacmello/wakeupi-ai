import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class UIColors:
    PRIMARY = (255, 100, 50)  # Laranja moderno
    SECONDARY = (50, 200, 255)  # Azul ciano
    SUCCESS = (100, 255, 100)  # Verde
    WARNING = (50, 200, 255)   # Amarelo
    DANGER = (50, 50, 255)     # Vermelho
    TEXT = (240, 240, 240)     # Quase branco
    BG_DARK = (20, 20, 25)     # Fundo escuro
    
class ModernUI:
    def __init__(self):
        self.font_family = cv2.FONT_HERSHEY_SIMPLEX
        self.panel_alpha = 0.60  # Mais transparente para ver o rosto
        
    def draw_modern_panel(self, frame: np.ndarray, stats: dict) -> np.ndarray:
        """
        Desenha um painel lateral moderno e compacto.
        """
        h, w = frame.shape[:2]
        panel_w = 200  # Mais estreito
        
        # Criar overlay para o painel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), UIColors.BG_DARK, -1)
        
        # Adicionar blur/transparência
        frame = cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0)
        
        # Configurações de layout
        margin_left = 20
        
        # Desenhar elementos
        y_pos = 40
        
        # Título menor e mais limpo
        self._draw_text(frame, "SLEEPARLET", (margin_left, y_pos), 0.6, UIColors.PRIMARY, 2)
        y_pos += 25
        self._draw_text(frame, "AI MONITORING", (margin_left, y_pos), 0.35, UIColors.SECONDARY, 1)
        
        # Linha divisória sutil
        y_pos += 15
        cv2.line(frame, (margin_left, y_pos), (panel_w - margin_left, y_pos), (60, 60, 70), 1)
        y_pos += 30
        
        # Status dos Olhos
        left_status = stats.get('left_status', 'Unknown')
        right_status = stats.get('right_status', 'Unknown')
        
        self._draw_status_row(frame, "Esq", left_status, margin_left, y_pos)
        y_pos += 25
        self._draw_status_row(frame, "Dir", right_status, margin_left, y_pos)
        
        y_pos += 20
        cv2.line(frame, (margin_left, y_pos), (panel_w - margin_left, y_pos), (60, 60, 70), 1)
        y_pos += 30
        
        # Métricas com fontes menores
        ear_val = stats.get('avg_ear', 0.0)
        self._draw_metric_bar(frame, "EAR", ear_val, 0.4, margin_left, y_pos, panel_w - 40)
        
        y_pos += 40
        blink_rate = stats.get('blink_rate', 0)
        self._draw_metric_value(frame, "Taxa (m)", f"{blink_rate:.1f}", margin_left, y_pos, panel_w)
        
        y_pos += 30
        blinks = stats.get('total_blinks', 0)
        self._draw_metric_value(frame, "Total", f"{blinks}", margin_left, y_pos, panel_w)
        
        # Rodapé discreto
        cv2.putText(frame, "v2.6 Minimal", (margin_left, h - 20), 
                   self.font_family, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
        
        return frame

    def _draw_text(self, img, text, pos, scale, color, thickness=1):
        cv2.putText(img, text, pos, self.font_family, scale, color, thickness, cv2.LINE_AA)

    def _draw_status_row(self, img, label, status, x, y):
        self._draw_text(img, label, (x, y), 0.4, UIColors.TEXT)
        
        color = UIColors.SUCCESS if status == "ABERTO" else UIColors.DANGER
        # Status abreviado ou ajustado
        status_display = status
        self._draw_text(img, status_display, (x + 40, y), 0.4, color, 1)
        
        # Indicador visual (círculo menor)
        circle_color = color
        indicator_x = x + 140
        cv2.circle(img, (indicator_x, y - 4), 4, circle_color, -1)
        if status == "ABERTO":
            cv2.circle(img, (indicator_x, y - 4), 6, (circle_color[0], circle_color[1], circle_color[2]), 1)

    def _draw_metric_bar(self, img, label, value, max_val, x, y, width):
        self._draw_text(img, f"{label}: {value:.3f}", (x, y - 8), 0.35, UIColors.TEXT)
        
        # Background da barra mais fino
        cv2.rectangle(img, (x, y), (x + width, y + 4), (40, 40, 50), -1)
        
        # Barra de valor
        fill_width = int((min(value, max_val) / max_val) * width)
        color = UIColors.SECONDARY
        if value < 0.2: color = UIColors.DANGER
        elif value < 0.25: color = UIColors.WARNING
            
        if fill_width > 0:
            cv2.rectangle(img, (x, y), (x + fill_width, y + 4), color, -1)

    def _draw_metric_value(self, img, label, value, x, y, panel_w):
        self._draw_text(img, label, (x, y), 0.35, (180, 180, 180))
        
        # Alinhar valor à direita do painel
        text_size = cv2.getTextSize(value, self.font_family, 0.5, 1)[0]
        value_x = panel_w - 20 - text_size[0]
        
        self._draw_text(img, value, (value_x, y), 0.5, UIColors.TEXT, 1)
