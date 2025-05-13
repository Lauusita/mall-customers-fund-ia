import os
from data_imputer import DataImputer
from data_preprocessor import DataPreprocessor
from model_evaluator import ModelEvaluator

def main():
    """
    Función principal que ejecuta el flujo completo del análisis:
    1. Imputación de datos faltantes
    2. Preprocesamiento de datos
    3. Evaluación de modelos con validación cruzada
    4. Comparación de modelos y selección del mejor
    """
    # Ruta al archivo de datos
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Mall_Customers-Missing Values.xlsx")
    
    # Fase 1: Imputación de datos
    imputer = DataImputer(path)
    imputer.load_data()
    data_imputed = imputer.impute_with_central_tendency()
    
    print("\nTÉCNICA DE IMPUTACIÓN | IMPUTACIÓN CON MEDIDA DE TENDENCIA CENTRAL")
    print(f"\n{data_imputed.head()}")
    
    # Fase 2: Preprocesamiento de datos
    preprocessor = DataPreprocessor(data_imputed)
    X, y = preprocessor.prepare_features_and_target()
    X_scaled = preprocessor.scale_features()
    
    # Fase 3: Evaluación de modelos
    evaluator = ModelEvaluator(X_scaled, y)
    
    # Evaluar diferentes configuraciones de MLP
    evaluator.evaluate_mlp_configurations()
    
    # Evaluar diferentes configuraciones de Regresión Logística
    evaluator.evaluate_logistic_regression()
    
    # Comparar modelos y obtener el mejor
    best_model, model_name = evaluator.compare_models()
    
    # Evaluar el rendimiento del mejor modelo
    evaluator.evaluate_model_performance(best_model, model_name)

if __name__ == "__main__":
    main()
