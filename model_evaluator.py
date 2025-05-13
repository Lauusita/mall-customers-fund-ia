import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ModelEvaluator:
    """
    Clase para evaluar diferentes modelos y configuraciones utilizando validación cruzada.
    Implementa evaluación para MLP y Regresión Logística.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_mlp_params = None
        self.best_mlp_score = 0
        self.best_lr_params = None
        self.best_lr_score = 0
        self.best_mlp = None
        self.best_lr = None
    
    def evaluate_mlp_configurations(self):
        """
        Evalúa diferentes configuraciones de MLP (Perceptrón Multicapa)
        utilizando validación cruzada de 5 pliegues.
        """
        print("\nEvaluando diferentes configuraciones de MLP...")
        
        # Definir configuraciones a evaluar para MLP
        configurations = [
            {'hidden_layer_sizes': (10,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001},
            {'hidden_layer_sizes': (20,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.001},
            {'hidden_layer_sizes': (10, 10), 'activation': 'tanh', 'solver': 'adam', 'alpha': 0.0001},
            {'hidden_layer_sizes': (20, 10), 'activation': 'tanh', 'solver': 'sgd', 'alpha': 0.01}
        ]
        
        # Configurar validación cruzada
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        best_score = 0
        best_params = None
        best_model = None
        
        # Evaluar cada configuración
        for config in configurations:
            print(f"Evaluando configuración: {config}")
            scores = []
            
            # Realizar validación cruzada
            for train_idx, test_idx in kf.split(self.X):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
                
                # Entrenar modelo
                mlp = MLPClassifier(max_iter=1000, random_state=42, **config)
                mlp.fit(X_train, y_train)
                
                # Evaluar modelo
                y_pred = mlp.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)
            
            # Calcular promedio de puntuaciones
            avg_score = np.mean(scores)
            print(f"Puntuación promedio: {avg_score:.4f}")
            
            # Actualizar mejor modelo si es necesario
            if avg_score > best_score:
                best_score = avg_score
                best_params = config
                # Entrenar modelo con todos los datos
                best_model = MLPClassifier(max_iter=1000, random_state=42, **config)
                best_model.fit(self.X, self.y)
        
        self.best_mlp_params = best_params
        self.best_mlp_score = best_score
        self.best_mlp = best_model
        
        print(f"\nMejor configuración MLP: {self.best_mlp_params}")
        print(f"Mejor puntuación MLP: {self.best_mlp_score:.4f}")
        
        return self.best_mlp_params, self.best_mlp_score
    
    def evaluate_logistic_regression(self):
        """
        Evalúa diferentes configuraciones de Regresión Logística
        utilizando validación cruzada de 5 pliegues.
        """
        print("\nEvaluando diferentes configuraciones de Regresión Logística...")
        
        # Definir configuraciones válidas para Regresión Logística
        configurations = [
            {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 200},
            {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg', 'max_iter': 200},
            {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 500},
            {'C': 10, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 200},
            {'C': 100, 'penalty': None, 'solver': 'lbfgs', 'max_iter': 500}
        ]
        
        # Configurar validación cruzada
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        best_score = 0
        best_params = None
        best_model = None
        
        # Evaluar cada configuración
        for config in configurations:
            print(f"Evaluando configuración: {config}")
            scores = []
            
            # Realizar validación cruzada
            for train_idx, test_idx in kf.split(self.X):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
                
                # Entrenar modelo
                lr = LogisticRegression(random_state=42, **config)
                lr.fit(X_train, y_train)
                
                # Evaluar modelo
                y_pred = lr.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)
            
            # Calcular promedio de puntuaciones
            avg_score = np.mean(scores)
            print(f"Puntuación promedio: {avg_score:.4f}")
            
            # Actualizar mejor modelo si es necesario
            if avg_score > best_score:
                best_score = avg_score
                best_params = config
                # Entrenar modelo con todos los datos
                best_model = LogisticRegression(random_state=42, **config)
                best_model.fit(self.X, self.y)
        
        self.best_lr_params = best_params
        self.best_lr_score = best_score
        self.best_lr = best_model
        
        print(f"\nMejor configuración Regresión Logística: {self.best_lr_params}")
        print(f"Mejor puntuación Regresión Logística: {self.best_lr_score:.4f}")
        
        return self.best_lr_params, self.best_lr_score
    
    def compare_models(self):
        """
        Compara los mejores modelos de MLP y Regresión Logística
        y devuelve el mejor de ellos.
        """
        print("\nComparando modelos...")
        
        if self.best_mlp_score > self.best_lr_score:
            print(f"El mejor modelo es MLP con una precisión de {self.best_mlp_score:.4f}")
            best_model = self.best_mlp
            model_name = "MLP"
        else:
            print(f"El mejor modelo es Regresión Logística con una precisión de {self.best_lr_score:.4f}")
            best_model = self.best_lr
            model_name = "Regresión Logística"
        
        return best_model, model_name
    
    def evaluate_model_performance(self, model, model_name):
        """
        Evalúa el rendimiento del modelo seleccionado, calculando
        métricas como precisión, sensibilidad y F1-score.
        """
        # Predecir con el mejor modelo
        y_pred = model.predict(self.X)
        
        # Calcular la precisión
        acc = accuracy_score(self.y, y_pred)
        
        # Calcular la matriz de confusión manualmente
        true_pos = sum((self.y == 1) & (y_pred == 1))
        true_neg = sum((self.y == 0) & (y_pred == 0))
        false_pos = sum((self.y == 0) & (y_pred == 1))
        false_neg = sum((self.y == 1) & (y_pred == 0))
        
        # Imprimir resultados
        print(f"\nResultados del modelo {model_name}:")
        print(f"Precisión: {acc:.4f}")
        print("\nMatriz de Confusión:")
        print(f"Verdaderos Positivos: {true_pos}")
        print(f"Verdaderos Negativos: {true_neg}")
        print(f"Falsos Positivos: {false_pos}")
        print(f"Falsos Negativos: {false_neg}")
        
        # Calcular métricas adicionales
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\nMétricas adicionales:")
        print(f"Precisión (Precision): {precision:.4f}")
        print(f"Sensibilidad (Recall): {recall:.4f}")
        print(f"Puntuación F1 (F1-Score): {f1:.4f}")
