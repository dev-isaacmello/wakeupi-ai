"""
Sistema de alerta para detecção de sonolência.
Gerencia estados de alerta e notificações sonoras.
"""
import time
import threading
import platform
import subprocess
from typing import Optional
from dataclasses import dataclass

from app.logger_config import logger
from app.config import config


@dataclass
class AlertState:
    """Estado atual do sistema de alertas."""
    active: bool = False
    flash_state: bool = False
    last_flash_time: float = 0.0
    last_beep_time: float = 0.0


class AlertSystem:
    """
    Sistema de alerta visual e sonoro para sonolência.
    
    Responsabilidade única: Gerenciar estados de alerta e notificações.
    A renderização visual é feita separadamente pelo AlertRenderer.
    """
    
    def __init__(self, alert_config: Optional[object] = None):
        """
        Inicializa o sistema de alertas.
        
        Args:
            alert_config: Configuração de alertas (usa config padrão se None)
        """
        self._config = alert_config or config.alert
        self._state = AlertState()
        self._lock = threading.Lock()
        logger.info("Sistema de alertas inicializado")
    
    @property
    def alert_active(self) -> bool:
        """Retorna se o alerta está ativo."""
        with self._lock:
            return self._state.active
    
    @property
    def flash_state(self) -> bool:
        """Retorna o estado atual do flash."""
        with self._lock:
            return self._state.flash_state
    
    def trigger_alert(self) -> None:
        """Ativa o estado de alerta."""
        with self._lock:
            if not self._state.active:
                self._state.active = True
                logger.warning("Alerta de sonolência ativado")
                # Tocar beep inicial em thread separada
                threading.Thread(target=self._play_beep, daemon=True).start()
    
    def reset_alert(self) -> None:
        """Desativa o estado de alerta."""
        with self._lock:
            if self._state.active:
                self._state.active = False
                self._state.flash_state = False
                logger.info("Alerta de sonolência desativado")
    
    def update(self) -> None:
        """
        Atualiza estados temporais do alerta (flashes, beeps periódicos).
        
        Deve ser chamado periodicamente para manter o estado atualizado.
        """
        with self._lock:
            if not self._state.active:
                return
            
            current_time = time.time()
            
            # Atualizar flash
            if current_time - self._state.last_flash_time >= self._config.flash_interval:
                self._state.flash_state = not self._state.flash_state
                self._state.last_flash_time = current_time
            
            # Tocar beep periódico
            if current_time - self._state.last_beep_time >= self._config.beep_interval:
                threading.Thread(target=self._play_beep, daemon=True).start()
                self._state.last_beep_time = current_time
    
    def _play_beep(self) -> None:
        """
        Toca som do sistema de forma agnóstica à plataforma.
        
        Trata erros silenciosamente para não interromper o fluxo principal.
        """
        try:
            system = platform.system()
            if system == "Windows":
                import winsound
                winsound.Beep(
                    self._config.beep_frequency,
                    self._config.beep_duration
                )
            elif system == "Darwin":
                subprocess.Popen(
                    ['say', 'beep'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Linux e outros sistemas Unix
                subprocess.Popen(
                    ['beep'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
        except ImportError as e:
            logger.warning(f"Módulo de som não disponível: {e}")
        except subprocess.SubprocessError as e:
            logger.warning(f"Erro ao executar comando de som: {e}")
        except Exception as e:
            logger.error(f"Erro inesperado ao tocar beep: {e}", exc_info=True)
