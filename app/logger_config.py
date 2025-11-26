"""
Configuração de logging estruturado para o SleepArlet.
"""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "sleeparlet",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configura e retorna um logger estruturado.
    
    Args:
        name: Nome do logger
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: String de formato personalizada (opcional)
    
    Returns:
        Logger configurado
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar handlers duplicados
    if logger.handlers:
        return logger
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


# Logger padrão da aplicação
logger = setup_logger()

