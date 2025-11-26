"""
Gerenciador de estado da aplicação.
Responsável por gerenciar estado de piscadas, alertas e estatísticas.
"""
import time
from collections import deque
from typing import Dict, Optional
from dataclasses import dataclass, field

from app.logger_config import logger
from app.config import config


@dataclass
class EyeState:
    """Estado atual dos olhos."""
    left_open: bool = True
    right_open: bool = True
    left_ear: float = 0.0
    right_ear: float = 0.0


@dataclass
class AppStatistics:
    """Estatísticas da aplicação."""
    blink_count: int = 0
    blink_rate: int = 0
    total_blinks: int = 0
    fps: int = 0
    alert: bool = False
    ear: float = 0.0


class StateManager:
    """
    Gerenciador de estado da aplicação.
    
    Responsabilidade única: Gerenciar estado de piscadas, alertas e estatísticas.
    """
    
    def __init__(self, detection_config: Optional[object] = None):
        """
        Inicializa o gerenciador de estado.
        
        Args:
            detection_config: Configuração de detecção (usa config padrão se None)
        """
        self._config = detection_config or config.detection
        
        self.blink_count: int = 0
        self.blink_times: deque = deque(maxlen=60)
        self.last_blink_time: float = time.time()
        self.eyes_closed_start: Optional[float] = None
        self.was_closed: bool = False
        
        self.last_stats: Dict = {
            'left_open': True,
            'right_open': True,
            'ear': 0.0,
            'blink_rate': 0,
            'total_blinks': 0,
            'alert': False,
            'fps': 0
        }
        
        logger.info("StateManager inicializado")
    
    def update_blink_state(
        self,
        user_left_closed: bool,
        user_right_closed: bool,
        current_time: float
    ) -> None:
        """
        Atualiza estado de piscadas.
        
        Args:
            user_left_closed: Se olho esquerdo do usuário está fechado
            user_right_closed: Se olho direito do usuário está fechado
            current_time: Tempo atual em segundos
        """
        eyes_closed = user_left_closed or user_right_closed
        
        # Lógica de contagem de piscadas
        if eyes_closed and not self.was_closed:
            if current_time - self.last_blink_time > self._config.blink_debounce:
                self.blink_count += 1
                self.blink_times.append(current_time)
                self.last_blink_time = current_time
                logger.debug(f"Piscada detectada (total: {self.blink_count})")
        
        self.was_closed = eyes_closed
    
    def update_drowsiness_state(
        self,
        eyes_closed: bool,
        current_time: float
    ) -> bool:
        """
        Atualiza estado de sonolência e retorna se alerta deve ser ativado.
        
        Args:
            eyes_closed: Se ambos os olhos estão fechados
            current_time: Tempo atual em segundos
        
        Returns:
            True se alerta deve ser ativado, False caso contrário
        """
        if eyes_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            
            duration = current_time - self.eyes_closed_start
            if duration > self._config.drowsiness_threshold:
                return True
        else:
            self.eyes_closed_start = None
        
        return False
    
    def calculate_blink_rate(self, current_time: float) -> int:
        """
        Calcula taxa de piscadas por minuto.
        
        Args:
            current_time: Tempo atual em segundos
        
        Returns:
            Taxa de piscadas por minuto
        """
        # Limpar piscadas antigas (mais de 60 segundos)
        while self.blink_times and (current_time - self.blink_times[0] > 60):
            self.blink_times.popleft()
        
        return len(self.blink_times)
    
    def update_stats(
        self,
        left_open: bool,
        right_open: bool,
        avg_ear: float,
        blink_rate: int,
        alert_active: bool,
        fps: int
    ) -> None:
        """
        Atualiza estatísticas globais.
        
        Args:
            left_open: Se olho esquerdo está aberto
            right_open: Se olho direito está aberto
            avg_ear: EAR médio
            blink_rate: Taxa de piscadas por minuto
            alert_active: Se alerta está ativo
            fps: FPS atual
        """
        self.last_stats = {
            'left_open': left_open,
            'right_open': right_open,
            'ear': float(avg_ear),
            'blink_rate': blink_rate,
            'total_blinks': self.blink_count,
            'alert': alert_active,
            'fps': int(fps)
        }
    
    def get_stats(self) -> Dict:
        """
        Retorna estatísticas atuais.
        
        Returns:
            Dicionário com estatísticas atuais
        """
        return self.last_stats.copy()

