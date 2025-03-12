import pandas as pd
import numpy as np
import os

# 🔹 Obtener la ruta absoluta de la carpeta principal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directorio donde está el script

# 🔹 Definir la ruta correcta de la carpeta /data/
DATA_DIR = os.path.join(BASE_DIR, "..", "data")  # Ir un nivel arriba y usar "data"

# 🔹 Crear la carpeta /data si no existe
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 🔹 Definir número de equipos a generar
n_equipos = 500  # Puedes ajustar este número según necesites

# 🔹 Generar variables aleatorias dentro de rangos realistas
np.random.seed(42)  # Para reproducibilidad

datos = {
    "Nota Estudiantes": np.random.randint(50, 100, n_equipos),
    "Nota Grupo": np.random.randint(50, 100, n_equipos),
    "Promedio Ponderado": np.random.randint(50, 100, n_equipos),
    "Variedad de Roles": np.random.randint(3, 9, n_equipos),  # Mínimo 3, máximo 9 roles
    "Coevaluación": np.random.randint(50, 100, n_equipos),
}

# 🔹 Definir la variable objetivo (Desempeño: 1 = Éxito, 0 = Fracaso)
datos["Desempeño"] = np.where(
    (datos["Promedio Ponderado"] > 75) & (datos["Variedad de Roles"] > 5), 1, 0
)

# 🔹 Convertir a DataFrame
df = pd.DataFrame(datos)

# 🔹 Guardar en la carpeta /data/ correctamente
csv_path = os.path.join("data", "datos_sinteticos.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Datos Sintéticos Generados y Guardados en {csv_path}")

