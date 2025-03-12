import pandas as pd
import numpy as np
import os

# 🔹 Definir roles de Belbin
ROLES_BELBIN = [
    "Coordinador", "Investigador de Recursos", "Trabajador", "Monitor Evaluador",
    "Impulsor", "Finalizador", "Especialista", "Cohesionador", "Creativo"
]

# 🔹 Definir la cantidad total de equipos
n_equipos = 500  # Se ajusta a la cantidad real de equipos en la base de datos

# 🔹 Crear carpeta /data si no existe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 🔹 Simulación de respuestas de estudiantes con número de equipo ingresado manualmente
np.random.seed(42)
estudiantes = []

for estudiante_id in range(n_equipos * 5):  # Suponiendo 5 estudiantes por equipo
    equipo_id = np.random.randint(0, n_equipos)  # El estudiante ingresa su número de equipo

    # Generar respuestas al test (cada sección tiene 10 puntos en total)
    respuestas = np.zeros(len(ROLES_BELBIN))

    for _ in range(7):  # Hay 7 secciones en el test
        distribucion_puntos = np.random.dirichlet(np.ones(len(ROLES_BELBIN))) * 10
        respuestas += distribucion_puntos.round()

    # Identificar el rol predominante
    rol_dominante = ROLES_BELBIN[np.argmax(respuestas)]

    # Generar una nota entre 0.0 y 5.0
    nota_estudiante = round(np.random.uniform(0, 5), 1)

    estudiantes.append({
        "Equipo ID": equipo_id,
        "Estudiante ID": estudiante_id,
        "Nota Estudiante": nota_estudiante,
        "Rol Dominante": rol_dominante,
        **dict(zip(ROLES_BELBIN, respuestas))  # Agregar todas las puntuaciones de roles
    })

# 🔹 Convertir a DataFrame
df_estudiantes = pd.DataFrame(estudiantes)

# 🔹 Calcular variedad de roles por equipo
df_variedad_roles = df_estudiantes.groupby("Equipo ID")["Rol Dominante"].nunique().reset_index()
df_variedad_roles.rename(columns={"Rol Dominante": "Variedad de Roles"}, inplace=True)

# 🔹 Calcular promedio de notas por equipo
df_promedio_notas = df_estudiantes.groupby("Equipo ID")["Nota Estudiante"].mean().reset_index()
df_promedio_notas.rename(columns={"Nota Estudiante": "Nota Grupo"}, inplace=True)

# 🔹 Generar evaluaciones de desempeño aleatorias (0.0 a 5.0)
df_evaluaciones = pd.DataFrame({
    "Equipo ID": np.arange(n_equipos),
    "Coevaluación": np.round(np.random.uniform(0, 5, n_equipos), 1)
})

# 🔹 Fusionar todos los datos en un solo DataFrame final
df_equipos = df_variedad_roles.merge(df_promedio_notas, on="Equipo ID").merge(df_evaluaciones, on="Equipo ID")

# 🔹 Definir la variable objetivo (Desempeño del equipo: 1 = Éxito, 0 = Fracaso)
df_equipos["Desempeño"] = np.where(
    (df_equipos["Nota Grupo"] > 3.5) & (df_equipos["Variedad de Roles"] > 4), 1, 0
)

# 🔹 Guardar datos de equipos y estudiantes en CSV
df_equipos.to_csv(os.path.join(DATA_DIR, "datos_equipos.csv"), index=False)
df_estudiantes.to_csv(os.path.join(DATA_DIR, "datos_estudiantes.csv"), index=False)

print(f"✅ Datos procesados y guardados en {DATA_DIR}")
