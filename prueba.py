import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Crear DataFrame con datos de ejemplo
data = {
    "Estudiante": [f"Est_{i+1}" for i in range(10)],
    "PI": [1.06, 0.40, 1.59, 0.60, 1.40, 1.20, 0.80, 1.00, 0.90, 1.10],
    "Motivacion": [3.5, 4.9, 4.1, 3.8, 3.2, 3.3, 4.0, 3.7, 3.9, 3.4]
}

df = pd.DataFrame(data)

# Gráfico de dispersión
plt.figure(figsize=(8, 6))
sns.scatterplot(x="PI", y="Motivacion", data=df, s=100, color="blue", alpha=0.7)
plt.xlabel("Índice de Persistencia (PI)")
plt.ylabel("Motivación Intrínseca (ESAA-2)")
plt.title("Relación entre Persistencia y Motivación Intrínseca")
plt.grid(True)

# Ajuste de regresión lineal
X = sm.add_constant(df["PI"])  # Agregar constante
y = df["Motivacion"]
modelo = sm.OLS(y, X).fit()
predicciones = modelo.predict(X)

# Graficar línea de regresión
plt.plot(df["PI"], predicciones, color="red", linewidth=2)
plt.show()

# Mostrar resumen del modelo
print(modelo.summary())

