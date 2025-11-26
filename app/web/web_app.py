"""
Aplicação web FastAPI para SleepArlet.
Responsável apenas por rotas HTTP e WebSocket.
"""
import json
import asyncio
import warnings
from typing import Dict

from fastapi import FastAPI, WebSocket, Request
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.logger_config import logger
from app.config import config
from app.detection.eye_detector import EyeDetector
from app.alert.alert_system import AlertSystem
from app.core.state_manager import StateManager
from app.core.video_processor import VideoProcessor

# Suprimir avisos do Protocol Buffer do MediaPipe
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='google.protobuf.symbol_database'
)

# Inicializar aplicação FastAPI
app = FastAPI(title="SleepArlet", version="3.0")

# Montar arquivos estáticos (caminho relativo à raiz do projeto)
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Instâncias globais (serão inicializadas no startup)
detector: EyeDetector = None
alert_system: AlertSystem = None
state_manager: StateManager = None
video_processor: VideoProcessor = None


@app.on_event("startup")
async def startup_event():
    """Inicializa componentes da aplicação ao iniciar."""
    global detector, alert_system, state_manager, video_processor
    
    logger.info("Inicializando aplicação SleepArlet...")
    
    # Criar instâncias com dependency injection
    detector = EyeDetector()
    alert_system = AlertSystem()
    state_manager = StateManager()
    video_processor = VideoProcessor(
        eye_detector=detector,
        alert_system=alert_system,
        state_manager=state_manager
    )
    
    logger.info("Aplicação inicializada com sucesso")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpa recursos ao encerrar aplicação."""
    from app.core.camera_manager import camera_manager
    
    logger.info("Encerrando aplicação...")
    camera_manager.release()
    logger.info("Aplicação encerrada")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    """
    Rota raiz - retorna página HTML principal.
    
    Args:
        request: Objeto Request do FastAPI
    
    Returns:
        Template HTML renderizado
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    """
    Rota de streaming de vídeo.
    
    Returns:
        StreamingResponse com frames JPEG codificados
    """
    if video_processor is None:
        logger.error("VideoProcessor não inicializado")
        return StreamingResponse(
            iter([b'']),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    return StreamingResponse(
        video_processor.process_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Endpoint WebSocket para telemetria em tempo real.
    
    Args:
        websocket: Conexão WebSocket
    """
    await websocket.accept()
    logger.info("Cliente WebSocket conectado")
    
    try:
        while True:
            if state_manager is None:
                await asyncio.sleep(0.1)
                continue
            
            # Enviar estado a ~30 FPS para economizar banda
            stats = state_manager.get_stats()
            await websocket.send_text(json.dumps(stats))
            await asyncio.sleep(config.web.websocket_update_interval)
    
    except WebSocketDisconnect:
        # Desconexão normal do cliente - não é um erro
        logger.debug("Cliente WebSocket desconectado normalmente")
    except Exception as e:
        # Erro real - logar com detalhes
        error_msg = str(e) if e else "Erro desconhecido"
        error_type = type(e).__name__
        logger.error(f"Erro no WebSocket ({error_type}): {error_msg}", exc_info=True)
    finally:
        logger.info("Cliente WebSocket desconectado")


def start() -> None:
    """
    Inicia o servidor FastAPI.
    
    Esta função é chamada quando o script é executado diretamente.
    """
    import uvicorn
    
    logger.info(
        f"Iniciando servidor em {config.web.host}:{config.web.port}"
    )
    logger.info(f"Acesse http://localhost:{config.web.port} no navegador")
    
    uvicorn.run(
        app,
        host=config.web.host,
        port=config.web.port,
        log_config=None  # Usamos nosso próprio logger
    )


if __name__ == "__main__":
    start()
