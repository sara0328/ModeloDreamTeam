import pandas as pd
import numpy as np
import os

# 🔹 Obtener la ruta absoluta de la carpeta principal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directorio donde está el script

# 🔹 Definir la ruta de la carpeta /data/ (ya existente)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")  # Ir un nivel arriba y usar "data"

# 🔹 Definir número de equipos a generar
n_equipos = 500  # Puedes ajustar este número según necesites

# 🔹 Generar variables aleatorias dentro de rangos realistas
np.random.seed(42)  # Para reproducibilidad

datos = {
    "Nota Estudiantes": np.random.randint(0, 5, n_equipos),  # Notas entre 0 y 5
    "Nota Grupo": np.random.randint(0, 5, n_equipos),
    "Promedio Ponderado": np.random.randint(0, 5, n_equipos),
    "Variedad de Roles": np.random.randint(3, 9, n_equipos),  # Mínimo 3, máximo 9 roles
    "Coevaluación": np.random.randint(0, 5, n_equipos),
}

# 🔹 Definir la variable objetivo (Desempeño: Entre 0 y 1)
datos["Desempeño"] = (
    (datos["Promedio Ponderado"] / 100) * 0.6 + 
    (datos["Variedad de Roles"] / 10) * 0.4
)

# 🔹 Convertir a DataFrame
df = pd.DataFrame(datos)

# 🔹 Guardar en la carpeta /data/ directamente
csv_path = os.path.join(DATA_DIR, "datos_sinteticos.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Datos Sintéticos Generados y Guardados en {csv_path}")
