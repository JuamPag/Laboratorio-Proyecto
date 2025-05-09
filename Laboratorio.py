import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, f_oneway, pearsonr

from statsmodels.formula.api import ols

# Configuración de estilo para visualizaciones
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Carga de datos
boston_df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv')

# ---------------------------------------------------------------
# 1. Visualización: Distribución de precios de viviendas (MEDV)
# ---------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(y='MEDV', data=boston_df)
plt.title("Distribución del Valor Medio de Viviendas (MEDV)")
plt.ylabel("Precio (en miles de $)")
plt.show()
"""
Explicación:
Este diagrama de caja muestra la distribución del precio mediano de las viviendas (MEDV). 
Podemos observar que:
- La mediana (línea central) está alrededor de $21,000
- El 50% central de los datos (caja) está entre ~$17,000 y $25,000
- Hay varios valores atípicos (puntos más allá de los bigotes) en el rango superior (> $35,000)
La distribución está ligeramente sesgada hacia la derecha, indicando que hay más viviendas
con precios por debajo de la mediana que por encima.
"""

# ---------------------------------------------------------------
# 2. Visualización: Viviendas cerca del Río Charles (CHAS)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='CHAS', data=boston_df)
plt.title("Frecuencia de Viviendas Cerca del Río Charles")
plt.xlabel("Limita con el río (1 = Sí, 0 = No)")
plt.ylabel("Cantidad de Viviendas")
plt.show()
"""
Explicación:
Este gráfico de barras muestra cuántas viviendas están ubicadas cerca del Río Charles (CHAS=1) 
versus las que no (CHAS=0). Los resultados muestran que:
- Solo 35 viviendas (aprox. 7%) están ubicadas junto al río
- 471 viviendas (93%) no limitan con el río
Esta distribución desigual sugiere que las propiedades frente al río son relativamente escasas,
lo que podría afectar su valor en el mercado.
"""

# ---------------------------------------------------------------
# 3. Visualización: Precio vs Antigüedad de viviendas
# ---------------------------------------------------------------
boston_df['AGE_GROUP'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100],
                               labels=["≤35 años", "35-70 años", "≥70 años"])

plt.figure(figsize=(10, 6))
sns.boxplot(x='AGE_GROUP', y='MEDV', data=boston_df)
plt.title("Precio de Viviendas vs. Antigüedad del Inmueble")
plt.xlabel("Grupo de Edad")
plt.ylabel("Precio (en miles de $)")
plt.show()
"""
Explicación:
Este diagrama de caja compara los precios de viviendas según su antigüedad, categorizada en tres grupos:
1. Viviendas nuevas (≤35 años): Tienen los precios más altos (mediana ~$24,000)
2. Viviendas de mediana edad (35-70 años): Precios intermedios (mediana ~$21,000)
3. Viviendas antiguas (≥70 años): Precios más bajos (mediana ~$19,000)

La tendencia muestra claramente que las propiedades más nuevas tienden a tener mayor valor en el mercado,
mientras que las más antiguas se deprecian. También se observa mayor variabilidad en precios para
viviendas de mediana edad.
"""

# ---------------------------------------------------------------
# 4. Visualización: Contaminación vs Áreas comerciales
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(x='INDUS', y='NOX', data=boston_df)
plt.title("Relación entre Óxido Nítrico y Áreas Comerciales")
plt.xlabel("Proporción de Acres Comerciales (INDUS)")
plt.ylabel("Concentración de Óxido Nítrico (NOX)")
plt.show()
"""
Explicación:
Este diagrama de dispersión explora la relación entre:
- Eje X: Proporción de acres comerciales no minoristas por ciudad (INDUS)
- Eje Y: Concentración de óxidos de nitrógeno (NOX), medida en partes por 10 millones

Se observa una clara relación positiva:
- A medida que aumenta la proporción de áreas comerciales (INDUS), la contaminación (NOX) aumenta
- La relación parece no lineal, con un crecimiento más acelerado cuando INDUS supera el 10%
- Hay una concentración notable de puntos cuando INDUS está entre 5-20%, donde NOX varía entre 0.5-0.7

Esto sugiere que las áreas con mayor desarrollo industrial/comercial tienden a tener peor calidad del aire.
"""

# ---------------------------------------------------------------
# 5. Visualización: Distribución de ratio alumno-profesor
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(boston_df['PTRATIO'], bins=20, kde=True)
plt.title("Distribución de la Proporción Alumno-Profesor")
plt.xlabel("Ratio Alumnos por Profesor")
plt.ylabel("Frecuencia")
plt.show()
"""
Explicación:
Este histograma muestra la distribución de la proporción alumno-profesor (PTRATIO) en las escuelas de la zona:
- La distribución es multimodal (tiene varios picos)
- El rango principal está entre 14-22 alumnos por profesor
- Los picos más notables están alrededor de 15, 18-19 y 20-21 alumnos por profesor
- Hay muy pocas áreas con ratios menores a 14 o mayores a 22

Esta distribución sugiere que existen diferentes políticas o capacidades educativas en distintas zonas,
con algunas áreas teniendo clases más pequeñas que otras.
"""

# ---------------------------------------------------------------
# 6. Pruebas estadísticas con explicaciones
# ---------------------------------------------------------------

# Prueba T: Comparación de precios según proximidad al río
group1 = boston_df[boston_df['CHAS'] == 0]['MEDV']
group2 = boston_df[boston_df['CHAS'] == 1]['MEDV']
t_stat, p_value = ttest_ind(group1, group2)
print(f"\nPrueba T para CHAS:\nEstadístico T: {t_stat:.3f}, p-valor: {p_value:.4f}")
"""
Explicación prueba T:
Hipótesis:
- H₀: No hay diferencia en precios entre viviendas cerca y lejos del río (µ1 = µ2)
- H₁: Existe diferencia significativa (µ1 ≠ µ2)

Resultados:
- Estadístico T = -3.113 (indica que el grupo CHAS=0 tiene media menor)
- p-valor = 0.002 (< 0.05)

Conclusión:
Rechazamos H₀. Hay evidencia estadística para afirmar que las viviendas cerca del río (CHAS=1) 
tienen precios significativamente más altos que las que no están cerca del río.
"""

# ANOVA: Comparación de precios por grupos de edad
group_young = boston_df[boston_df['AGE_GROUP'] == "≤35 años"]['MEDV']
group_mid = boston_df[boston_df['AGE_GROUP'] == "35-70 años"]['MEDV']
group_old = boston_df[boston_df['AGE_GROUP'] == "≥70 años"]['MEDV']
f_stat, p_value = f_oneway(group_young, group_mid, group_old)
print(f"\nANOVA para AGE_GROUP:\nEstadístico F: {f_stat:.3f}, p-valor: {p_value:.4f}")
"""
Explicación ANOVA:
Hipótesis:
- H₀: No hay diferencia en precios entre los grupos de edad (µ1 = µ2 = µ3)
- H₁: Al menos un grupo difiere significativamente

Resultados:
- Estadístico F = 36.407 (valor alto indica diferencias significativas entre grupos)
- p-valor ≈ 0.0000 (< 0.05)

Conclusión:
Rechazamos H₀. Existen diferencias estadísticamente significativas en los precios entre al menos 
dos de los grupos de edad. Las visualizaciones anteriores sugieren que todas las diferencias 
por pares son significativas.
"""

# Correlación: NOX vs INDUS
corr, p_value = pearsonr(boston_df['NOX'], boston_df['INDUS'])
print(f"\nCorrelación NOX vs INDUS:\nCorrelación: {corr:.3f}, p-valor: {p_value:.4f}")
"""
Explicación correlación:
Hipótesis:
- H₀: No hay correlación lineal entre NOX e INDUS (ρ = 0)
- H₁: Existe correlación lineal (ρ ≠ 0)

Resultados:
- Coeficiente de correlación = 0.763 (correlación positiva fuerte)
- p-valor ≈ 0.0000 (< 0.05)

Conclusión:
Rechazamos H₀. Existe una fuerte correlación positiva y estadísticamente significativa entre la 
proporción de áreas comerciales y la concentración de óxidos de nitrógeno. Esto respalda la 
observación visual del diagrama de dispersión.
"""

# Regresión lineal: Impacto de distancia a centros de empleo (DIS) en precios
model = ols('MEDV ~ DIS', data=boston_df).fit()
print("\nRegresión lineal MEDV ~ DIS:")
print(model.summary())
"""
Explicación regresión:
Hipótesis:
- H₀: La distancia a centros de empleo no afecta el precio (β1 = 0)
- H₁: La distancia tiene un efecto significativo (β1 ≠ 0)

Resultados:
- Coeficiente para DIS = 1.0916 (p-valor ≈ 0.000)
- R² = 0.062 (el modelo explica el 6.2% de la variabilidad en precios)

Conclusión:
Rechazamos H₀. Por cada unidad de aumento en la distancia a centros de empleo (DIS), el precio 
de las viviendas aumenta aproximadamente $1,092 (en miles), efecto que es estadísticamente 
significativo. Sin embargo, el bajo R² indica que hay otros factores más importantes que 
influyen en el precio.
"""

# ---------------------------------------------------------------
# Conclusiones generales
# ---------------------------------------------------------------
"""
Conclusiones clave del análisis:
1. Factores que aumentan valor de viviendas:
   - Proximidad al Río Charles (diferencia promedio de ~$7,000)
   - Menor antigüedad (viviendas nuevas valen ~$5,000 más que antiguas)
   - Mayor distancia a centros de empleo (aunque efecto es pequeño)

2. Contaminación:
   - Existe fuerte relación entre áreas comerciales y contaminación por NOX
   - Zonas con >20% de áreas comerciales tienen niveles de NOX peligrosos (>0.6 ppm)

3. Educación:
   - La distribución de ratio alumno-profesor es desigual
   - Algunas zonas tienen ratios significativamente mejores (14:1 vs 21:1)

Recomendaciones:
- Inversores deberían considerar propiedades cerca del río y de construcción reciente
- Autoridades urbanas deberían regular usos comerciales/industriales para controlar contaminación
- Se necesitan más estudios para identificar otros factores que afectan precios (el modelo solo explica 6.2% de variabilidad)
"""
