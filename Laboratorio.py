import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, f_oneway, pearsonr

from statsmodels.formula.api import ols

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

boston_df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv')

plt.figure(figsize=(8, 6))
sns.boxplot(y='MEDV', data=boston_df)
plt.title("Distribución del Valor Medio de Viviendas (MEDV)")
plt.ylabel("Precio (en miles de $)")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='CHAS', data=boston_df)
plt.title("Frecuencia de Viviendas Cerca del Río Charles")
plt.xlabel("Limita con el río (1 = Sí, 0 = No)")
plt.ylabel("Cantidad de Viviendas")
plt.show()

boston_df['AGE_GROUP'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, 100], labels=["≤35 años", "35-70 años", "≥70 años"])

plt.figure(figsize=(10, 6))
sns.boxplot(x='AGE_GROUP', y='MEDV', data=boston_df)
plt.title("Precio de Viviendas vs. Antigüedad del Inmueble")
plt.xlabel("Grupo de Edad")
plt.ylabel("Precio (en miles de $)")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='INDUS', y='NOX', data=boston_df)
plt.title("Relación entre Óxido Nítrico y Áreas Comerciales")
plt.xlabel("Proporción de Acres Comerciales (INDUS)")
plt.ylabel("Concentración de Óxido Nítrico (NOX)")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(boston_df['PTRATIO'], bins=20, kde=True)
plt.title("Distribución de la Proporción Alumno-Profesor")
plt.xlabel("Ratio Alumnos por Profesor")
plt.ylabel("Frecuencia")
plt.show()

group1 = boston_df[boston_df['CHAS'] == 0]['MEDV']
group2 = boston_df[boston_df['CHAS'] == 1]['MEDV']

t_stat, p_value = ttest_ind(group1, group2)
print(f"Prueba T para CHAS:\nEstadístico T: {t_stat:.3f}, p-valor: {p_value:.4f}")

group_young = boston_df[boston_df['AGE_GROUP'] == "≤35 años"]['MEDV']
group_mid = boston_df[boston_df['AGE_GROUP'] == "35-70 años"]['MEDV']
group_old = boston_df[boston_df['AGE_GROUP'] == "≥70 años"]['MEDV']

f_stat, p_value = f_oneway(group_young, group_mid, group_old)
print(f"\nANOVA para AGE_GROUP:\nEstadístico F: {f_stat:.3f}, p-valor: {p_value:.4f}")

corr, p_value = pearsonr(boston_df['NOX'], boston_df['INDUS'])
print(f"\nCorrelación NOX vs INDUS:\nCorrelación: {corr:.3f}, p-valor: {p_value:.4f}")

model = ols('MEDV ~ DIS', data=boston_df).fit()
print("\nRegresión lineal MEDV ~ DIS:")
print(model.summary())