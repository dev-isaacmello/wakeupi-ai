"""
Detector de olhos usando MediaPipe Face Mesh.
Responsável por detectar landmarks, calcular EAR e determinar estado dos olhos.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional
from collections import deque

from app.logger_config import logger
from app.config import config


class EyeDetector:
    """
    Detector de olhos usando MediaPipe Face Mesh.
    
    Responsabilidade: Detectar landmarks, calcular EAR e estado (aberto/fechado).
    A renderização visual é feita separadamente pelo EyeRenderer.
    """
    
    # Índices MediaPipe Iris (segundo documentação oficial)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Contornos para extração de imagem (região de interesse)
    LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    def __init__(self, detection_config: Optional[object] = None):
        """
        Inicializa o detector de olhos.
        
        Args:
            detection_config: Configuração de detecção (usa config padrão se None)
        """
        self._config = detection_config or config.detection
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        self.left_ear_history = deque(maxlen=self._config.ear_smoothing_frames)
        self.right_ear_history = deque(maxlen=self._config.ear_smoothing_frames)
        
        self.left_ear_baseline: Optional[float] = None
        self.right_ear_baseline: Optional[float] = None
        
        logger.info("EyeDetector inicializado")
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
        """
        Processa o frame e retorna EARs e landmarks.
        
        Args:
            frame: Frame BGR do OpenCV
        
        Returns:
            Tupla (left_ear, right_ear, landmarks)
            - left_ear: EAR do olho esquerdo (MediaPipe) ou None
            - right_ear: EAR do olho direito (MediaPipe) ou None
            - landmarks: Array numpy com landmarks ou None
        """
        h, w = frame.shape[:2]
        
        # Redimensionar para performance
        target_width = 480
        if w > target_width:
            scale = target_width / w
            new_h = int(h * scale)
            process_frame = cv2.resize(
                frame,
                (target_width, new_h),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            process_frame = frame
        
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.face_mesh.process(rgb_frame)
        except Exception as e:
            logger.error(f"Erro ao processar frame com MediaPipe: {e}", exc_info=True)
            self.left_ear_history.clear()
            self.right_ear_history.clear()
            return None, None, None
        
        if not results.multi_face_landmarks:
            self.left_ear_history.clear()
            self.right_ear_history.clear()
            return None, None, None
        
        # Converter landmarks para escala original
        try:
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
        
        except (IndexError, AttributeError) as e:
            logger.warning(f"Erro ao processar landmarks: {e}")
            self.left_ear_history.clear()
            self.right_ear_history.clear()
            return None, None, None
        except Exception as e:
            logger.error(f"Erro inesperado ao processar frame: {e}", exc_info=True)
            self.left_ear_history.clear()
            self.right_ear_history.clear()
            return None, None, None
    
    def is_eye_closed(
        self,
        ear: float,
        baseline: Optional[float],
        frame: np.ndarray,
        landmarks: np.ndarray,
        eye_indices: list,
        eye_contour: list,
        force_deep_learning: bool = False
    ) -> bool:
        """
        Determina se um olho específico está fechado.
        
        Args:
            ear: Valor EAR do olho
            baseline: Baseline EAR do olho (opcional)
            frame: Frame BGR completo
            landmarks: Landmarks do rosto
            eye_indices: Índices dos pontos do olho
            eye_contour: Contorno do olho
            force_deep_learning: Forçar uso de deep learning (não implementado ainda)
        
        Returns:
            True se o olho está fechado, False caso contrário
        """
        if ear is None:
            return False
        
        # Se EAR está muito baixo, definitivamente está fechado
        if ear < 0.15:
            return True
        
        # Lógica de Threshold Adaptativo
        threshold = self._config.ear_threshold
        
        # Usar baseline se disponível
        if baseline is not None:
            # Threshold adaptativo baseado no baseline (65% do baseline)
            adaptive_threshold = baseline * 0.65
            # Garantir que o threshold não fique muito alto nem muito baixo
            threshold = max(0.18, min(adaptive_threshold, 0.28))
        elif len(self.left_ear_history) > 3:
            # Fallback para histórico recente
            try:
                avg_ear = np.mean(list(self.left_ear_history))
                threshold = max(0.20, min(avg_ear * 0.70, 0.26))
            except Exception as e:
                logger.warning(f"Erro ao calcular threshold adaptativo: {e}")
                threshold = self._config.ear_threshold
        
        closed_by_ear = ear < threshold
        
        # TODO: Implementar validação por Deep Learning quando necessário
        # if self._config.use_deep_learning and force_deep_learning:
        #     ...
        
        return closed_by_ear
    
    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calcula Eye Aspect Ratio.
        
        Args:
            eye_points: Array numpy com 6 pontos do olho
        
        Returns:
            Valor EAR calculado
        """
        try:
            # Pontos verticais
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            # Ponto horizontal
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            
            if h == 0:
                return 0.0
            
            return (v1 + v2) / (2.0 * h)
        except (IndexError, ValueError) as e:
            logger.warning(f"Erro ao calcular EAR: {e}")
            return 0.0
    
    def _update_baselines(self, left: float, right: float) -> None:
        """
        Atualiza baselines dinamicamente usando os maiores valores (olhos abertos).
        
        Args:
            left: EAR do olho esquerdo
            right: EAR do olho direito
        """
        if self.left_ear_baseline is None:
            self.left_ear_baseline = left
        else:
            self.left_ear_baseline = max(self.left_ear_baseline * 0.99, left)
        
        if self.right_ear_baseline is None:
            self.right_ear_baseline = right
        else:
            self.right_ear_baseline = max(self.right_ear_baseline * 0.99, right)
    
    def _smooth_ear(self, history: deque, value: float) -> float:
        """
        Suaviza valor EAR usando histórico.
        
        Args:
            history: Deque com histórico de valores EAR
            value: Valor EAR atual
        
        Returns:
            Média do histórico incluindo o valor atual
        """
        history.append(value)
        try:
            return float(np.mean(history))
        except Exception as e:
            logger.warning(f"Erro ao calcular média do EAR: {e}")
            return value
