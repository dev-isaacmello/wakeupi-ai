# Sistema de Detecção de Sonolência

Sistema Python de alta precisão para detectar quando você está com os olhos fechados por mais de 1.3 segundos, emitindo alertas visuais para te acordar.

## Características

- **Detecção precisa de olhos** usando MediaPipe Face Mesh
- **Cálculo de EAR (Eye Aspect Ratio)** para determinar estado dos olhos
- **Monitoramento de tempo** com olhos fechados
- **Sistema de alerta visual** com popup e flash vermelho
- **Contador de piscadas** (blink rate) em tempo real
- **Interface gráfica** com visualização da webcam e indicadores

## Requisitos

- Python 3.8, 3.9, 3.10 ou 3.11 (Python 3.12 pode funcionar, mas Python 3.13 ainda não é suportado pelo MediaPipe)
- Webcam
- Windows/Linux/MacOS

**⚠️ IMPORTANTE:** Se você está usando Python 3.13, você precisará usar Python 3.11 ou anterior. O MediaPipe ainda não tem builds disponíveis para Python 3.13.

## Instalação

1. Clone ou baixe este repositório

2. **Verifique sua versão do Python:**
```bash
python --version
```
Se você estiver usando Python 3.13, você precisará instalar Python 3.11 ou anterior. Você pode baixar Python 3.11 em [python.org](https://www.python.org/downloads/).

3. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
```

4. Ative o ambiente virtual:
   - **Windows PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows CMD:**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux/MacOS:**
     ```bash
     source venv/bin/activate
     ```

5. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso

Execute o script principal:

```bash
python main.py
```

### Controles

- **'q'**: Sair do programa
- O sistema inicia automaticamente a captura da webcam

### Funcionalidades

- **Detecção em tempo real**: O sistema detecta seu rosto e monitora seus olhos continuamente
- **Alerta visual**: Quando seus olhos ficam fechados por mais de 1.3 segundos, um alerta vermelho piscante aparece na tela
- **Informações na tela**:
  - Status dos olhos (ABERTOS/FECHADOS)
  - EAR médio (Eye Aspect Ratio)
  - Tempo com olhos fechados
  - Taxa de piscadas por minuto
  - Contador total de piscadas
  - FPS do sistema

## Como Funciona

### Eye Aspect Ratio (EAR)

O sistema usa o algoritmo EAR para determinar se os olhos estão abertos ou fechados:

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

Onde p1-p6 são pontos específicos dos olhos detectados pelo MediaPipe Face Mesh.

- **EAR alto** (> 0.25): Olhos abertos
- **EAR baixo** (< 0.25): Olhos fechados

### Detecção de Sonolência

1. O sistema monitora continuamente o estado dos olhos
2. Quando os olhos fecham, um timer é iniciado
3. Se os olhos permanecem fechados por mais de 1.3 segundos, o alerta é disparado
4. O alerta visual (flash vermelho) continua até que os olhos sejam abertos novamente

### Detecção de Piscadas

- O sistema detecta transições de olhos abertos para fechados
- Calcula a taxa de piscadas por minuto baseado nas últimas piscadas
- Mantém um histórico das últimas 60 piscadas

## Ajustes

Você pode ajustar os parâmetros no código:

### `main.py`

- `drowsiness_threshold`: Tempo em segundos com olhos fechados para disparar alerta (padrão: 1.3)

### `eye_detector.py`

- `EAR_THRESHOLD`: Threshold para determinar se olho está fechado (padrão: 0.25)

### `alert_system.py`

- `flash_interval`: Intervalo entre flashes do alerta em segundos (padrão: 0.3)

## Estrutura do Projeto

```
sleeparlet/
├── main.py              # Script principal
├── eye_detector.py      # Classe de detecção de olhos
├── alert_system.py      # Sistema de alertas visuais
├── requirements.txt     # Dependências
└── README.md           # Este arquivo
```

## Dependências

- `opencv-python`: Processamento de imagem e captura de vídeo
- `mediapipe`: Detecção facial e landmarks
- `numpy`: Cálculos numéricos

## Notas

- Certifique-se de ter boa iluminação para melhor detecção
- Mantenha o rosto visível para a câmera
- O sistema funciona melhor com uma pessoa por vez na frente da câmera
- Para melhor precisão, ajuste o threshold EAR se necessário

## Troubleshooting

**Problema**: Rosto não detectado
- **Solução**: Melhore a iluminação e certifique-se de que seu rosto está visível

**Problema**: Falsos positivos (alerta quando olhos estão abertos)
- **Solução**: Aumente o `EAR_THRESHOLD` em `eye_detector.py`

**Problema**: Não detecta olhos fechados
- **Solução**: Diminua o `EAR_THRESHOLD` em `eye_detector.py`

**Problema**: Webcam não abre
- **Solução**: Verifique se a webcam não está sendo usada por outro programa

**Problema**: Erro "No matching distribution found for mediapipe"
- **Solução**: Você provavelmente está usando Python 3.13. Instale Python 3.11 ou anterior. O MediaPipe ainda não suporta Python 3.13.

## Licença

Este projeto é fornecido como está, para uso pessoal e educacional.

