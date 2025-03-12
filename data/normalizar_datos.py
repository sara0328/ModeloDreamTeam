import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


# Obtener la ruta absoluta de la carpeta donde está el script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")  # Ir un nivel arriba a la carpeta 'data'

# Definir la ruta completa del archivo CSV
csv_path = os.path.join(DATA_DIR, "datos_sinteticos.csv")

# Cargar el archivo
df = pd.read_csv(csv_path)

print("✅ Archivo cargado correctamente")


# 🔹 Cargar los datos
csv_path = os.path.join("data", "datos_sinteticos.csv")
df = pd.read_csv(csv_path)

# 🔹 Definir las columnas numéricas a normalizar
columnas_numericas = ["Nota Estudiantes", "Nota Grupo", "Promedio Ponderado", "Coevaluación", "Variedad de Roles"]

# 🔹 Inicializar el MinMaxScaler
scaler = MinMaxScaler()

# 🔹 Aplicar la normalización
df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

# 🔹 Guardar el dataset procesado
csv_norm_path = os.path.join("data", "datos_sinteticos_normalizados.csv")
df.to_csv(csv_norm_path, index=False)

print(f"✅ Datos normalizados y guardados en {csv_norm_path}")
