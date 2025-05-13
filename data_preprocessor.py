import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    Clase para el preprocesamiento de datos antes del modelado.
    Incluye selección de características y escalado de datos.
    """
    def __init__(self, data):
        self.data = data
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = StandardScaler()
    
    def prepare_features_and_target(self):
        """
        Selecciona las características relevantes y la variable objetivo.
        Variables relevantes: Age, Annual Income (k$), Spending Score (1-100)
        Variable objetivo: Suitable for a bank loan
        """
        # Seleccionar solo las variables relevantes para la predicción
        self.X = self.data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        self.y = self.data["Suitable for a bank loan"]
        return self.X, self.y
    
    def scale_features(self):
        """
        Escala las características utilizando StandardScaler para mejorar
        el rendimiento de los modelos.
        """
        # Escalar las características para mejorar el rendimiento de los modelos
        self.X_scaled = self.scaler.fit_transform(self.X)
        return self.X_scaled
