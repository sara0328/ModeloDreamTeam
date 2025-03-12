import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 📌 Ruta del archivo con datos normalizados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "datos_normalizados.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "modelo_regresion.pkl")

# 📌 Cargar los datos normalizados
df = pd.read_csv(DATA_PATH)
print("✅ Archivo de datos normalizados cargado correctamente.")

# 📌 Definir variables independientes (X) y la variable objetivo (y)
X = df.drop(columns=["Desempeño"])  # Todas las columnas excepto la de desempeño
y = df["Desempeño"]  # Variable objetivo

# 📌 Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 📌 Crear el modelo de regresión logística
modelo = LogisticRegression()

# 📌 Entrenar el modelo
modelo.fit(X_train, y_train)
print("✅ Modelo entrenado correctamente.")

# 📌 Hacer predicciones
y_pred = modelo.predict(X_test)

# 📌 Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Precisión del modelo: {accuracy:.4f}")
print("\n📌 Reporte de clasificación:\n", classification_report(y_test, y_pred))

# 📌 Guardar el modelo entrenado
with open(MODEL_PATH, "wb") as modelo_file:
    pickle.dump(modelo, modelo_file)

print(f"✅ Modelo guardado en {MODEL_PATH}")
