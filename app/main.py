"""
Ponto de entrada principal do SleepArlet.
"""
from app.web.web_app import start

if __name__ == "__main__":
    # Inicia o servidor FastAPI
    print("Iniciando SleepArlet v3.0 (Web Interface)...")
    print("Acesse http://localhost:8000 no seu navegador.")
    start()
