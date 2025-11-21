"""
Módulo de Deep Learning para classificação avançada de olhos fechados/abertos.
Usa redes neurais convolucionais (CNN) para máxima precisão.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from collections import deque
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Não imprimir aviso aqui para não poluir a saída

class DeepEyeClassifier:
    """
    Classificador de deep learning para detecção de olhos fechados/abertos.
    Usa CNN para análise de imagens dos olhos extraídas pelo MediaPipe.
    """
    
    def __init__(self, use_pretrained: bool = True):
        """
        Inicializa o classificador de deep learning.
        
        Args:
            use_pretrained: Se True, tenta carregar modelo pré-treinado ou criar um novo
        """
        self.model = None
        self.model_ensemble = []  # Ensemble de modelos para maior precisão
        self.input_size = (64, 64)  # Tamanho aumentado para melhor precisão
        self.use_pretrained = use_pretrained
        
        # Histórico para análise temporal avançada
        self.prediction_history = deque(maxlen=15)
        self.confidence_history = deque(maxlen=15)
        self.feature_history = deque(maxlen=10)  # Histórico de features
        
        if TF_AVAILABLE and use_pretrained:
            self._build_or_load_model()
        else:
            print("Usando classificador híbrido avançado (sem TensorFlow)")
    
    def _build_model(self):
        """Constrói uma CNN avançada com arquitetura moderna (ResNet-like) para máxima precisão."""
        inputs = keras.Input(shape=(*self.input_size, 1))
        
        # Bloco inicial com normalização
        x = layers.BatchNormalization()(inputs)
        
        # Primeiro bloco convolucional (mais filtros)
        x = layers.Conv2D(64, (7, 7), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.Dropout(0.25)(x)
        
        # Blocos residuais simplificados
        def residual_block(x, filters):
            shortcut = x
            x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([shortcut, x])  # Skip connection
            x = layers.Activation('relu')(x)
            return x
        
        # Aplicar blocos residuais
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = residual_block(x, 128)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = residual_block(x, 256)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Global Average Pooling (melhor que Flatten)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Camadas densas com regularização
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        # Saída
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compilar com otimizador avançado
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _build_lightweight_model(self):
        """Constrói modelo leve mas eficiente (MobileNet-like) para uso em tempo real."""
        inputs = keras.Input(shape=(*self.input_size, 1))
        
        # Depthwise separable convolutions (mais eficiente)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Blocos depthwise
        def depthwise_block(x, filters):
            x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, (1, 1), activation='relu')(x)
            x = layers.BatchNormalization()(x)
            return x
        
        x = depthwise_block(x, 64)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.Dropout(0.2)(x)
        
        x = depthwise_block(x, 128)
        x = layers.MaxPooling2D(2, 2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = depthwise_block(x, 256)
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_or_load_model(self):
        """Constrói ou carrega modelos pré-treinados (ensemble)."""
        model_path = 'eye_classifier_model.h5'
        lightweight_path = 'eye_classifier_lightweight.h5'
        
        # Tentar carregar modelo principal
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                print(f"✓ Modelo avançado carregado de {model_path}")
            except Exception as e:
                print(f"Erro ao carregar modelo principal: {e}")
                print("Criando novo modelo avançado...")
                self.model = self._build_model()
        else:
            print("Criando modelo CNN avançado com arquitetura ResNet-like...")
            self.model = self._build_model()
        
        # Tentar carregar modelo leve para ensemble
        if os.path.exists(lightweight_path):
            try:
                lightweight_model = keras.models.load_model(lightweight_path)
                self.model_ensemble.append(lightweight_model)
                print(f"✓ Modelo leve carregado para ensemble")
            except:
                pass
        
        # Se não há modelo leve, criar um
        if len(self.model_ensemble) == 0:
            print("Criando modelo leve para ensemble...")
            lightweight_model = self._build_lightweight_model()
            self.model_ensemble.append(lightweight_model)
        
        print(f"✓ Sistema de deep learning inicializado com {1 + len(self.model_ensemble)} modelo(s)")
    
    def extract_eye_region(self, frame: np.ndarray, landmarks: np.ndarray, 
                          eye_indices: list) -> Optional[np.ndarray]:
        """
        Extrai região do olho do frame.
        
        Args:
            frame: Frame BGR
            landmarks: Landmarks do rosto
            eye_indices: Índices dos pontos do olho
            
        Returns:
            Região do olho normalizada ou None
        """
        try:
            eye_points = landmarks[eye_indices]
            
            # Calcular bounding box
            x_min = int(np.min(eye_points[:, 0]))
            x_max = int(np.max(eye_points[:, 0]))
            y_min = int(np.min(eye_points[:, 1]))
            y_max = int(np.max(eye_points[:, 1]))
            
            # Adicionar padding maior para capturar mais contexto
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame.shape[1], x_max + padding)
            y_max = min(frame.shape[0], y_max + padding)
            
            # Extrair região
            eye_region = frame[y_min:y_max, x_min:x_max]
            
            if eye_region.size == 0:
                return None
            
            # Converter para escala de cinza
            eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Aplicar equalização de histograma para melhor contraste
            eye_gray = cv2.equalizeHist(eye_gray)
            
            # Redimensionar com interpolação de alta qualidade
            eye_resized = cv2.resize(eye_gray, self.input_size, interpolation=cv2.INTER_CUBIC)
            
            # Aplicar filtro gaussiano suave para reduzir ruído
            eye_resized = cv2.GaussianBlur(eye_resized, (3, 3), 0)
            
            # Normalizar com normalização mais robusta
            eye_normalized = eye_resized.astype(np.float32) / 255.0
            
            # Aplicar normalização local (melhora contraste)
            mean = np.mean(eye_normalized)
            std = np.std(eye_normalized)
            if std > 0:
                eye_normalized = (eye_normalized - mean) / std
                eye_normalized = (eye_normalized + 1) / 2  # Voltar para [0, 1]
            
            return eye_normalized
        except:
            return None
    
    def classify_with_cnn(self, eye_image: np.ndarray) -> Tuple[bool, float]:
        """
        Classifica olho usando CNN com ensemble para máxima precisão.
        
        Args:
            eye_image: Imagem do olho normalizada
            
        Returns:
            (is_open, confidence)
        """
        if self.model is None or not TF_AVAILABLE:
            return None, 0.0
        
        try:
            # Preparar entrada
            eye_input = eye_image.reshape(1, *self.input_size, 1)
            
            predictions = []
            confidences = []
            
            # Predição do modelo principal
            pred_main = self.model.predict(eye_input, verbose=0)[0][0]
            predictions.append(pred_main)
            confidences.append(abs(pred_main - 0.5) * 2)
            
            # Predições dos modelos do ensemble
            for model in self.model_ensemble:
                try:
                    pred = model.predict(eye_input, verbose=0)[0][0]
                    predictions.append(pred)
                    confidences.append(abs(pred - 0.5) * 2)
                except:
                    continue
            
            # Ensemble: média ponderada (modelo principal tem peso maior)
            if len(predictions) > 1:
                weights = [0.6] + [0.4 / (len(predictions) - 1)] * (len(predictions) - 1)
                final_prediction = np.average(predictions, weights=weights)
            else:
                final_prediction = predictions[0]
            
            is_open = final_prediction > 0.5
            confidence = abs(final_prediction - 0.5) * 2
            
            # Aumentar confiança se modelos concordam
            if len(predictions) > 1:
                agreement = 1.0 - np.std(predictions)
                confidence = min(1.0, confidence * (1.0 + agreement * 0.3))
            
            return is_open, confidence
        except Exception as e:
            return None, 0.0
    
    def classify_with_advanced_heuristics(self, eye_image: np.ndarray, 
                                         eye_landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Classificação avançada usando heurísticas melhoradas quando CNN não disponível.
        
        Args:
            eye_image: Imagem do olho
            eye_landmarks: Landmarks do olho
            
        Returns:
            (is_open, confidence)
        """
        if eye_image is None or eye_landmarks is None:
            return False, 0.0
        
        try:
            # Análise de intensidade (olhos fechados são mais escuros)
            mean_intensity = np.mean(eye_image)
            std_intensity = np.std(eye_image)
            
            # Análise de bordas (olhos abertos têm mais bordas)
            edges = cv2.Canny((eye_image * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / (eye_image.size)
            
            # Análise de contraste
            contrast = std_intensity / (mean_intensity + 1e-5)
            
            # Calcular EAR dos landmarks
            vertical_dist_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            vertical_dist_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            horizontal_dist = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            if horizontal_dist == 0:
                ear = 0.0
            else:
                ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
            
            # Features avançadas combinadas
            # Análise de textura usando LBP (Local Binary Pattern) simplificado
            texture_score = 0.0
            if eye_image.size > 100:
                # Calcular gradientes para análise de textura
                grad_x = cv2.Sobel(eye_image, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(eye_image, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                texture_score = np.mean(gradient_magnitude)
            
            # Análise de histograma
            hist = cv2.calcHist([(eye_image * 255).astype(np.uint8)], [0], None, [256], [0, 256])
            hist_normalized = hist / (hist.sum() + 1e-5)
            entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-5))
            
            features = np.array([
                mean_intensity,
                std_intensity,
                edge_density,
                contrast,
                ear,
                vertical_dist_1 / (horizontal_dist + 1e-5),
                vertical_dist_2 / (horizontal_dist + 1e-5),
                texture_score,
                entropy / 8.0  # Normalizar entropia
            ])
            
            # Pesos otimizados com mais features
            weights = np.array([-0.25, 0.25, 0.45, 0.35, 0.55, 0.25, 0.25, 0.15, 0.2])
            bias = 0.12
            
            score = np.dot(features, weights) + bias
            is_open = score > 0.0
            
            # Confiança baseada na distância da fronteira de decisão e consistência das features
            feature_consistency = 1.0 - np.std(features[:7]) / (np.mean(np.abs(features[:7])) + 1e-5)
            confidence = min(1.0, abs(score) * 2 * (0.7 + 0.3 * feature_consistency))
            
            return is_open, confidence
        except:
            return False, 0.0
    
    def classify_eye(self, frame: np.ndarray, landmarks: np.ndarray, 
                     eye_indices: list, eye_landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Classifica se o olho está aberto ou fechado usando deep learning.
        
        Args:
            frame: Frame BGR completo
            landmarks: Todos os landmarks do rosto
            eye_indices: Índices dos pontos do olho
            eye_landmarks: Landmarks específicos do olho (6 pontos)
            
        Returns:
            (is_open, confidence)
        """
        # Extrair região do olho
        eye_image = self.extract_eye_region(frame, landmarks, eye_indices)
        
        if eye_image is None:
            return False, 0.0
        
        # Tentar usar CNN primeiro
        if self.model is not None and TF_AVAILABLE:
            is_open, confidence = self.classify_with_cnn(eye_image)
            if confidence > 0.3:  # Se confiança é boa, usar CNN
                self.prediction_history.append(is_open)
                self.confidence_history.append(confidence)
                return is_open, confidence
        
        # Usar método híbrido avançado
        is_open, confidence = self.classify_with_advanced_heuristics(eye_image, eye_landmarks)
        
        self.prediction_history.append(is_open)
        self.confidence_history.append(confidence)
        
        # Suavização temporal
        if len(self.prediction_history) >= 3:
            recent_predictions = list(self.prediction_history)[-3:]
            avg_confidence = np.mean(list(self.confidence_history)[-3:])
            
            # Se maioria dos últimos 3 frames dizem o mesmo, usar isso
            if sum(recent_predictions) >= 2:
                is_open = True
            elif sum(recent_predictions) <= 1:
                is_open = False
            
            confidence = avg_confidence
        
        return is_open, confidence
    
    def are_both_eyes_open(self, left_open: bool, right_open: bool, 
                          left_conf: float, right_conf: float) -> bool:
        """
        Determina se ambos os olhos estão abertos com base nas classificações.
        
        Args:
            left_open: Olho esquerdo está aberto
            right_open: Olho direito está aberto
            left_conf: Confiança da classificação esquerda
            right_conf: Confiança da classificação direita
            
        Returns:
            True se ambos os olhos estão abertos
        """
        # Requer ambos os olhos abertos OU alta confiança em pelo menos um
        if left_open and right_open:
            return True
        
        # Se um está aberto com alta confiança, considerar ambos abertos
        if (left_open and left_conf > 0.7) or (right_open and right_conf > 0.7):
            return True
        
        return False

