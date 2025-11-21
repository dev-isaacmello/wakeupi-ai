import os
import warnings
import cv2
import time
import numpy as np
from collections import deque
from eye_detector import EyeDetector
from alert_system import AlertSystem
from ui_modern import ModernUI

# Configurações de ambiente
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

class App:
    """
    Classe principal da aplicação SleepArlet.
    Orquestra a captura de vídeo, detecção e interface.
    """
    
    def __init__(self):
        self.setup_system()
        self.state = {
            'blink_count': 0,
            'blink_times': deque(maxlen=60),
            'last_blink_time': time.time(),
            'eyes_closed_start': None,
            'is_drowsy': False
        }
        
    def setup_system(self):
        """Inicializa componentes do sistema."""
        self.detector = EyeDetector(use_deep_learning=True)
        self.alerts = AlertSystem()
        self.ui = ModernUI()
        
        # Configuração da Câmera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível acessar a webcam.")
            
    def run(self):
        """Loop principal da aplicação."""
        print("SleepArlet Iniciado - Pressione 'q' para sair")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # 1. Processamento
            left_ear, right_ear, landmarks = self.detector.process_frame(frame)
            
            # 2. Lógica de Detecção
            stats = self.process_detection(frame, left_ear, right_ear, landmarks)
            
            # 3. Renderização
            # Círculos nos olhos
            frame = self.detector.draw_debug_circles(frame, landmarks, left_ear, right_ear)
            # UI Moderna
            frame = self.ui.draw_modern_panel(frame, stats)
            # Overlay de Alerta
            frame = self.alerts.create_alert_overlay(frame)
            
            # 4. Exibição
            cv2.imshow('SleepArlet v2.6', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()

    def process_detection(self, frame, left_ear, right_ear, landmarks):
        """Processa estados lógicos baseados nos dados brutos."""
        if left_ear is None or right_ear is None:
            return {'left_status': 'N/A', 'right_status': 'N/A', 'avg_ear': 0.0, 'blink_rate': 0, 'total_blinks': 0}

        # Verificar estado de cada olho
        left_closed = self.detector.is_eye_closed(
            left_ear, self.detector.left_ear_baseline, 
            frame, landmarks, 
            self.detector.LEFT_EYE_INDICES, self.detector.LEFT_EYE_CONTOUR
        )
        
        right_closed = self.detector.is_eye_closed(
            right_ear, self.detector.right_ear_baseline,
            frame, landmarks,
            self.detector.RIGHT_EYE_INDICES, self.detector.RIGHT_EYE_CONTOUR
        )
        
        eyes_closed = left_closed and right_closed
        
        # Atualizar Piscadas
        self._update_blinks(left_closed or right_closed) # Conta se qualquer um fechar
        
        # Atualizar Sonolência
        self._update_drowsiness(eyes_closed)
        
        return {
            'left_status': 'FECHADO' if left_closed else 'ABERTO',
            'right_status': 'FECHADO' if right_closed else 'ABERTO',
            'avg_ear': (left_ear + right_ear) / 2,
            'blink_rate': self._calculate_blink_rate(),
            'total_blinks': self.state['blink_count']
        }

    def _update_blinks(self, is_closed):
        """Lógica de contagem de piscadas."""
        now = time.time()
        # Se fechou agora e não estava fechado antes (com debounce)
        if is_closed and (now - self.state['last_blink_time'] > 0.2):
             # Lógica simplificada para exemplo; idealmente precisa de transição estado aberto->fechado
             # Aqui vamos assumir que o detector cuida da suavização
             pass 
             
        # Detecção de borda de subida simples para o exemplo:
        if is_closed and not getattr(self, '_was_closed', False):
            if now - self.state['last_blink_time'] > 0.15: # Debounce
                self.state['blink_count'] += 1
                self.state['blink_times'].append(now)
                self.state['last_blink_time'] = now
        
        self._was_closed = is_closed

    def _update_drowsiness(self, eyes_closed):
        """Gerencia estado de alerta de sonolência."""
        if eyes_closed:
            if self.state['eyes_closed_start'] is None:
                self.state['eyes_closed_start'] = time.time()
            
            duration = time.time() - self.state['eyes_closed_start']
            if duration > 1.3: # Threshold de sonolência
                self.alerts.trigger_alert()
        else:
            self.state['eyes_closed_start'] = None
            self.alerts.reset_alert()

    def _calculate_blink_rate(self):
        """Calcula piscadas por minuto."""
        now = time.time()
        # Limpar piscadas antigas
        while self.state['blink_times'] and (now - self.state['blink_times'][0] > 60):
            self.state['blink_times'].popleft()
            
        return len(self.state['blink_times'])

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.alerts.reset_alert()

if __name__ == "__main__":
    try:
        app = App()
        app.run()
    except Exception as e:
        print(f"Erro fatal: {e}")
