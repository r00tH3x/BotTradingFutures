# BUAT FILE BARU: ml_enhancer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class MLSignalEnhancer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame, indicators: dict) -> np.array:
        """Prepare features for ML model"""
        features = []
        
        # Technical indicators
        features.extend([
            indicators.get('rsi', np.array([50]))[-1],
            indicators.get('macd_hist', np.array([0]))[-1],
            indicators.get('stoch_k', np.array([50]))[-1],
        ])
        
        # Price action features
        recent_closes = df['close'].tail(5).values
        features.extend([
            np.mean(recent_closes),
            np.std(recent_closes),
            (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        ])
        
        # Volume features
        recent_volumes = df['volume'].tail(5).values
        features.extend([
            np.mean(recent_volumes),
            recent_volumes[-1] / np.mean(recent_volumes)
        ])
        
        return np.array(features).reshape(1, -1)
    
    def enhance_signal(self, confidence: float, features: np.array) -> float:
        """Enhance signal confidence using ML"""
        if not self.is_trained:
            return confidence
        
        try:
            features_scaled = self.scaler.transform(features)
            ml_confidence = self.model.predict_proba(features_scaled)[0][1]
            
            # Weighted average of technical analysis and ML
            enhanced_confidence = (confidence * 0.7) + (ml_confidence * 100 * 0.3)
            return min(enhanced_confidence, 100)
            
        except Exception as e:
            return confidence
# BUAT FILE BARU: ml_enhancer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class MLSignalEnhancer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame, indicators: dict) -> np.array:
        """Prepare features for ML model"""
        features = []
        
        # Technical indicators
        features.extend([
            indicators.get('rsi', np.array([50]))[-1],
            indicators.get('macd_hist', np.array([0]))[-1],
            indicators.get('stoch_k', np.array([50]))[-1],
        ])
        
        # Price action features
        recent_closes = df['close'].tail(5).values
        features.extend([
            np.mean(recent_closes),
            np.std(recent_closes),
            (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        ])
        
        # Volume features
        recent_volumes = df['volume'].tail(5).values
        features.extend([
            np.mean(recent_volumes),
            recent_volumes[-1] / np.mean(recent_volumes)
        ])
        
        return np.array(features).reshape(1, -1)
    
    def enhance_signal(self, confidence: float, features: np.array) -> float:
        """Enhance signal confidence using ML"""
        if not self.is_trained:
            return confidence
        
        try:
            features_scaled = self.scaler.transform(features)
            ml_confidence = self.model.predict_proba(features_scaled)[0][1]
            
            # Weighted average of technical analysis and ML
            enhanced_confidence = (confidence * 0.7) + (ml_confidence * 100 * 0.3)
            return min(enhanced_confidence, 100)
            
        except Exception as e:
            return confidence
