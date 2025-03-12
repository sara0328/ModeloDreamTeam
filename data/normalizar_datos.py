import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# 📌 Ruta absoluta del directorio de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datos_sinteticos.csv")

# 📌 Cargar los datos sintéticos generados
df = pd.read_csv(DATA_PATH)
print("✅ Archivo cargado correctamente")

# 📌 Definir las columnas numéricas a normalizar
columnas_numericas = ["Nota Estudiantes", "Nota Grupo", "Promedio Ponderado", "Coevaluación", "Variedad de Roles"]

# 📌 Aplicar MinMaxScaler para normalizar entre 0 y 1
scaler = MinMaxScaler()
df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

# 📌 Guardar los datos normalizados en un nuevo archivo
normalized_csv_path = os.path.join(BASE_DIR, "datos_normalizados.csv")
df.to_csv(normalized_csv_path, index=False)

print(f"✅ Datos normalizados correctamente y guardados en {normalized_csv_path}")
