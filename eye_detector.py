import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional
from collections import deque
from deep_eye_classifier import DeepEyeClassifier

class EyeDetector:
    """
    Detector de olhos usando MediaPipe Face Mesh.
    Responsabilidade: Detectar landmarks, calcular EAR e estado (aberto/fechado).
    """
    
    # Índices MediaPipe Iris (segundo documentação oficial)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Contornos para extração de imagem (região de interesse)
    LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # Configurações
    EAR_THRESHOLD = 0.20
    EAR_SMOOTHING_FRAMES = 5
    
    def __init__(self, use_deep_learning: bool = False):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        self.left_ear_history = deque(maxlen=self.EAR_SMOOTHING_FRAMES)
        self.right_ear_history = deque(maxlen=self.EAR_SMOOTHING_FRAMES)
        
        self.left_ear_baseline = None
        self.right_ear_baseline = None
        
        self.use_deep_learning = use_deep_learning
        self.deep_classifier = None
        if use_deep_learning:
            try:
                self.deep_classifier = DeepEyeClassifier(use_pretrained=True)
            except Exception as e:
                print(f"Deep Learning indisponível: {e}")
                self.use_deep_learning = False

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
        """Processa o frame e retorna EARs e landmarks."""
        h, w = frame.shape[:2]
        
        # Redimensionar para performance se necessário
        process_frame = frame
        scale = 1.0
        if w > 640:
            scale = 640.0 / w
            new_h = int(h * scale)
            process_frame = cv2.resize(frame, (640, new_h))
            
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            self.left_ear_history.clear()
            self.right_ear_history.clear()
            return None, None, None
            
        # Converter landmarks para escala original
        face_landmarks = results.multi_face_landmarks[0]
        h_orig, w_orig = frame.shape[:2]
        landmarks = np.array([
            [lm.x * w_orig, lm.y * h_orig] for lm in face_landmarks.landmark
        ])
        
        # Calcular EAR
        left_ear = self._calculate_ear(landmarks[self.LEFT_EYE_INDICES])
        right_ear = self._calculate_ear(landmarks[self.RIGHT_EYE_INDICES])
        
        # Atualizar baselines
        self._update_baselines(left_ear, right_ear)
        
        # Suavização
        left_smooth = self._smooth_ear(self.left_ear_history, left_ear)
        right_smooth = self._smooth_ear(self.right_ear_history, right_ear)
        
        return left_smooth, right_smooth, landmarks

    def is_eye_closed(self, ear: float, baseline: Optional[float], 
                      frame: np.ndarray, landmarks: np.ndarray, 
                      eye_indices: list, eye_contour: list) -> bool:
        """Determina se um olho específico está fechado."""
        if ear is None: return False
        
        # Lógica de Threshold Adaptativo
        threshold = self.EAR_THRESHOLD
        if baseline is not None:
            threshold = max(0.10, min(baseline * 0.65, 0.25))
        elif len(self.left_ear_history) > 3: # Fallback para histórico recente
             threshold = max(0.15, min(np.mean(list(self.left_ear_history)) * 0.70, 0.22))

        closed_by_ear = ear < threshold
        
        # Validação por Deep Learning (se habilitado e ambíguo)
        if self.use_deep_learning and self.deep_classifier:
            # Se EAR está próximo do threshold (zona de incerteza)
            if abs(ear - threshold) < 0.05:
                eye_landmarks = landmarks[eye_indices]
                is_open, conf = self.deep_classifier.classify_eye(
                    frame, landmarks, eye_contour, eye_landmarks
                )
                if conf > 0.6:
                    return not is_open
                    
        return closed_by_ear

    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """Calcula Eye Aspect Ratio."""
        # Pontos verticais
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        # Ponto horizontal
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if h == 0: return 0.0
        return (v1 + v2) / (2.0 * h)

    def _update_baselines(self, left: float, right: float):
        """Atualiza baselines dinamicamente usando os maiores valores (olhos abertos)."""
        # Simplificado: usa percentil superior do histórico
        pass # Lógica movida para processamento direto ou mantida simples
        
        # Para manter consistência com a versão anterior que funcionava bem:
        if self.left_ear_baseline is None: self.left_ear_baseline = left
        else: self.left_ear_baseline = max(self.left_ear_baseline * 0.99, left) # Decaimento lento
        
        if self.right_ear_baseline is None: self.right_ear_baseline = right
        else: self.right_ear_baseline = max(self.right_ear_baseline * 0.99, right)

    def _smooth_ear(self, history: deque, value: float) -> float:
        history.append(value)
        return np.mean(history)

    def draw_debug_circles(self, frame: np.ndarray, landmarks: np.ndarray, 
                          left_ear: float, right_ear: float) -> np.ndarray:
        """Desenha visualização dos olhos."""
        if landmarks is None: return frame
        frame_copy = frame.copy()
        
        for indices, ear, baseline in [
            (self.LEFT_EYE_INDICES, left_ear, self.left_ear_baseline), 
            (self.RIGHT_EYE_INDICES, right_ear, self.right_ear_baseline)
        ]:
            points = landmarks[indices].astype(int)
            center = np.mean(points, axis=0).astype(int)
            
            # Cor baseada no estado
            threshold = self.EAR_THRESHOLD
            if baseline: threshold = baseline * 0.65
            
            color = (0, 0, 255) if ear < threshold else (0, 255, 200)
            
            # Raio dinâmico
            w = np.max(points[:, 0]) - np.min(points[:, 0])
            radius = int(w * 0.7)
            
            cv2.circle(frame_copy, tuple(center), radius, color, 2)
            
        return frame_copy
