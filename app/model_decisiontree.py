import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 📌 Ajustar según tu estructura de archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "datos_normalizados.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "modelo_arbol_regresion.pkl")

# 1. Cargar los datos normalizados
df = pd.read_csv(DATA_PATH)
print("✅ Archivo de datos normalizados cargado correctamente.")

# 2. Definir variables independientes (X) y la variable objetivo (y)
#    Asegúrate de reemplazar "Desempeño" por el nombre real de tu columna de desempeño
X = df.drop(columns=["Desempeño"])   # Todas las columnas excepto la de desempeño
y = df["Desempeño"]                  # Variable objetivo (entre 0 y 1)

# 3. Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

modelo = DecisionTreeRegressor(
    max_depth=3,       # Limitamos la profundidad
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# 5. Entrenar el modelo
modelo.fit(X_train, y_train)
print("✅ Árbol de Decisión (Regresión) entrenado correctamente.")

# 6. Hacer predicciones en el set de prueba
y_pred = modelo.predict(X_test)

# 7. Evaluar el modelo con métricas de regresión
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 MSE (Error Cuadrático Medio): {mse:.4f}")
print(f"📊 MAE (Error Absoluto Medio): {mae:.4f}")
print(f"📊 R² (Coef. de Determinación): {r2:.4f}")

# 8. Guardar el modelo entrenado
with open(MODEL_PATH, "wb") as modelo_file:
    pickle.dump(modelo, modelo_file)

print(f"✅ Modelo de Árbol de Decisión guardado en: {MODEL_PATH}")

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

