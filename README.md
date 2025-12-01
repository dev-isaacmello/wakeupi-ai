# SleepArlet - Sistema de Detecção de Sonolência

Sistema de monitoramento em tempo real que detecta quando os olhos permanecem fechados por mais de 0.8 segundos, emitindo alertas visuais e sonoros para prevenir acidentes por sonolência.

**Autor:** Isaac Mello  
**Versão:** 3.0.0

---

## 📋 Visão Geral

O SleepArlet utiliza visão computacional e deep learning para monitorar o estado dos olhos através da webcam, calculando o Eye Aspect Ratio (EAR) e aplicando modelos de classificação para determinar com precisão quando os olhos estão fechados.

### Tecnologias Utilizadas

- **FastAPI**: Framework web moderno e rápido
- **MediaPipe Face Mesh**: Detecção facial e landmarks precisos
- **OpenCV**: Processamento de imagem e captura de vídeo
- **TensorFlow/Keras**: Modelos de deep learning para classificação avançada (opcional)
- **NumPy**: Cálculos numéricos otimizados
- **WebSocket**: Comunicação em tempo real com o frontend

---

## 🚀 Requisitos

### Sistema

- **Python**: 3.8, 3.9, 3.10 ou 3.11
- **Webcam**: Funcional e acessível
- **Sistema Operacional**: Windows, Linux ou macOS

> **⚠️ Nota:** Python 3.13 não é suportado pelo MediaPipe. Use Python 3.11 ou anterior.

### Dependências

Todas as dependências estão listadas em `requirements.txt`:

- `opencv-python >= 4.8.0`
- `mediapipe >= 0.10.0`
- `numpy >= 1.24.0`
- `tensorflow >= 2.13.0` (opcional, para deep learning)
- `fastapi >= 0.100.0`
- `uvicorn >= 0.22.0`
- `jinja2 >= 3.1.0`
- `websockets >= 11.0`

---

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/dev-isaacmello/sleeparlet-ai
cd sleeparlet-ai
```

### 2. Verifique a versão do Python

```bash
python --version
```

Se necessário, instale Python 3.11 ou anterior em [python.org](https://www.python.org/downloads/).

### 3. Crie e ative o ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## ▶️ Uso

Execute o script principal:

```bash
python main.py
```

O servidor será iniciado e você verá:

```
Iniciando SleepArlet v3.0 (Web Interface)...
Acesse http://localhost:8000 no seu navegador.
```

### Interface Web

Acesse `http://localhost:8000` no seu navegador para usar a interface web moderna que exibe em tempo real:

- **Status dos olhos**: ABERTO/FECHADO para cada olho
- **EAR médio**: Eye Aspect Ratio calculado
- **Taxa de piscadas**: Piscadas por minuto
- **Total de piscadas**: Contador acumulado
- **Gráfico EAR**: Visualização em tempo real do nível de abertura dos olhos
- **FPS**: Taxa de quadros por segundo

### Alerta de Sonolência

Quando os olhos permanecem fechados por **0.8 segundos**, o sistema dispara:

- **Alerta visual**: Overlay vermelho pulsante na tela
- **Alerta sonoro**: Beep do sistema
- **Mensagem**: "VOCE DORMIU!!!! ACORDE AGORA!!!"

O alerta permanece ativo até que os olhos sejam abertos novamente.

---

## 🏗️ Arquitetura

O projeto segue princípios SOLID e melhores práticas Python, com arquitetura modular e bem organizada:

### Estrutura de Diretórios

```
sleeparlet-ai/
├── app/                          # Pacote principal da aplicação
│   ├── __init__.py
│   ├── main.py                   # Entry point interno
│   ├── config.py                 # Configurações centralizadas
│   ├── logger_config.py          # Configuração de logging
│   │
│   ├── core/                     # Módulos core
│   │   ├── __init__.py
│   │   ├── camera_manager.py     # Gerenciador Singleton de câmera
│   │   ├── state_manager.py      # Gerenciador de estado da aplicação
│   │   └── video_processor.py    # Processador de vídeo
│   │
│   ├── detection/                 # Módulos de detecção
│   │   ├── __init__.py
│   │   ├── eye_detector.py       # Detector de olhos (EAR)
│   │   └── deep_eye_classifier.py # Classificador de deep learning
│   │
│   ├── alert/                     # Sistema de alertas
│   │   ├── __init__.py
│   │   ├── alert_system.py        # Lógica de alertas
│   │   └── alert_renderer.py      # Renderização de alertas
│   │
│   ├── rendering/                 # Renderização visual
│   │   ├── __init__.py
│   │   └── eye_renderer.py       # Renderizador visual de olhos
│   │
│   └── web/                       # Aplicação web
│       ├── __init__.py
│       └── web_app.py             # Rotas FastAPI e WebSocket
│
├── static/                        # Arquivos estáticos
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── dashboard.js
│
├── templates/                     # Templates HTML
│   └── index.html
│
├── main.py                       # Entry point principal
├── requirements.txt              # Dependências
├── README.md                     # Documentação
└── AUDITORIA_CHECKLIST.md        # Checklist de auditoria
```

### Princípios de Design

- **Single Responsibility**: Cada classe tem uma única responsabilidade
- **Dependency Injection**: Dependências injetadas via construtores
- **Separation of Concerns**: Renderização separada da lógica de negócio
- **Singleton Pattern**: `CameraManager` garante uma única instância de câmera
- **Modularidade**: Código organizado em módulos temáticos

---

## 🔧 Funcionamento Técnico

### Eye Aspect Ratio (EAR)

O sistema calcula o EAR usando 6 pontos específicos dos olhos detectados pelo MediaPipe:

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

**Interpretação:**
- **EAR > 0.25**: Olhos abertos
- **EAR < 0.25**: Olhos fechados
- **EAR < 0.15**: Definitivamente fechado (detecção imediata)

### Threshold Adaptativo

O sistema utiliza um threshold adaptativo baseado no baseline individual:

- Calcula o baseline dinâmico dos olhos abertos
- Ajusta o threshold para 65% do baseline
- Mantém limites entre 0.18 e 0.28 para evitar falsos positivos

### Deep Learning (Opcional)

Quando habilitado, o sistema utiliza modelos CNN para validação em casos ambíguos:

- Modelo principal: Arquitetura ResNet-like
- Ativado apenas quando EAR está próximo do threshold (zona de incerteza)
- Fallback para heurísticas avançadas quando TensorFlow não está disponível

### Otimizações de Performance

- Processamento em resolução reduzida (480px)
- Deep learning apenas quando necessário (a cada 0.5s)
- Modificação in-place de frames para reduzir cópias
- MediaPipe com refinamento de landmarks para precisão
- WebSocket com atualização a ~30 FPS para economizar banda

---

## ⚙️ Configurações

Todas as configurações estão centralizadas em `app/config.py`:

### Configurações de Câmera

```python
CameraConfig(
    device_id=0,
    width=640,
    height=480,
    fps=30,
    buffer_size=1
)
```

### Configurações de Detecção

```python
DetectionConfig(
    ear_threshold=0.25,
    ear_smoothing_frames=5,
    use_deep_learning=False,
    deep_learning_check_interval=0.5,
    drowsiness_threshold=0.8,  # segundos
    blink_debounce=0.15        # segundos
)
```

### Configurações de Alerta

```python
AlertConfig(
    flash_interval=0.2,      # segundos
    beep_interval=0.5,       # segundos
    beep_frequency=1000,     # Hz
    beep_duration=200        # ms
)
```

### Configurações Web

```python
WebConfig(
    host="0.0.0.0",
    port=8000,
    websocket_update_interval=0.033  # ~30 FPS
)
```

Para modificar configurações, edite `app/config.py` ou crie uma instância customizada de `AppConfig`.

---

## 🐛 Solução de Problemas

### Rosto não detectado

- **Causa**: Iluminação insuficiente ou rosto fora do campo de visão
- **Solução**: Melhore a iluminação e posicione-se centralmente na frente da câmera

### Falsos positivos (alerta com olhos abertos)

- **Causa**: Threshold muito baixo ou baseline incorreto
- **Solução**: Aumente `ear_threshold` em `app/config.py` (ex: 0.27 ou 0.28)

### Não detecta olhos fechados

- **Causa**: Threshold muito alto
- **Solução**: Diminua `ear_threshold` em `app/config.py` (ex: 0.22 ou 0.23)

### Webcam não abre

- **Causa**: Webcam em uso por outro programa ou permissões
- **Solução**: Feche outros programas que usam a webcam e verifique permissões do sistema

### Erro ao instalar MediaPipe

- **Causa**: Versão do Python incompatível (Python 3.13)
- **Solução**: Instale Python 3.11 ou anterior

### FPS muito baixo

- **Causa**: Processamento pesado ou hardware limitado
- **Solução**: O sistema já está otimizado. Se necessário, desabilite deep learning em `app/config.py`:
  ```python
  DetectionConfig(use_deep_learning=False)
  ```

### Erro de importação após reorganização

- **Causa**: Imports antigos ou ambiente virtual não atualizado
- **Solução**: Certifique-se de estar usando a versão mais recente do código e reinstale as dependências:
  ```bash
  pip install -r requirements.txt
  ```

---

## 📝 Notas de Uso

- **Iluminação**: Mantenha boa iluminação frontal para melhor detecção
- **Posicionamento**: Mantenha o rosto visível e centralizado na câmera
- **Ambiente**: Funciona melhor com uma pessoa por vez na frente da câmera
- **Ajuste fino**: Ajuste as configurações em `app/config.py` conforme necessário para seu ambiente
- **Logs**: O sistema utiliza logging estruturado. Configure o nível em `app/logger_config.py`

---

## 🔍 Logging

O sistema utiliza logging estruturado configurado em `app/logger_config.py`. Os logs incluem:

- **DEBUG**: Informações detalhadas de debug
- **INFO**: Informações gerais de operação
- **WARNING**: Avisos sobre problemas não críticos
- **ERROR**: Erros que requerem atenção

Para ajustar o nível de log, modifique `setup_logger()` em `app/logger_config.py`.

---

## 📄 Licença

Este projeto é de uso pessoal e educacional.

**Desenvolvido por Isaac Mello - AI Engineer**

---
