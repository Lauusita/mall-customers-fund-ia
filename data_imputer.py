import pandas as pd
import os

class DataImputer:
    """
    Clase para la imputación de datos faltantes en el dataset.
    Utiliza medidas de tendencia central para rellenar valores nulos.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
    def load_data(self):
        """Carga los datos desde un archivo Excel."""
        self.data = pd.read_excel(self.file_path)
        return self.data
    
    def impute_with_central_tendency(self):
        """
        Imputa valores faltantes utilizando medidas de tendencia central.
        - Media para la edad
        - Mediana para ingresos anuales y puntuación de gasto
        """
        # Extraer columnas relevantes
        age = self.data["Age"]
        annual_income = self.data["Annual Income (k$)"]
        spending_score = self.data["Spending Score (1-100)"]
        
        # Calcular medidas de tendencia central
        promedio_edad = round(age.mean())
        promedio_income = round(annual_income.median())
        promedio_spending_score = round(spending_score.median())
        
        # Definir valores por defecto para imputación
        default_values = {
            "Age": promedio_edad,
            "Annual Income (k$)": promedio_income,
            "Spending Score (1-100)": promedio_spending_score
        }
        
        # Imputar valores faltantes
        self.data.fillna(value=default_values, inplace=True)
        return self.data
