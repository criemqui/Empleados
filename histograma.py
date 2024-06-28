import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Especifica la ruta donde se encuentra el archivo CSV
file_path = 'C:/Users/eduar/Desktop/EMPLEADOS/EmployeesData.csv'  # Asegúrate de reemplazar con la ruta correcta

# Cargar el archivo CSV en un DataFrame de pandas
data = pd.read_csv(file_path)

# Verificar si existen valores faltantes en el DataFrame
missing_values = data.isna().sum()

# Mostrar las columnas con valores faltantes, si las hay
print("Valores faltantes por columna:")
print(missing_values[missing_values > 0])

# Convertir la columna 'LeaveOrNot' a etiquetas categóricas
data['LeaveOrNot'] = data['LeaveOrNot'].replace({1: 'Leave', 0: 'Not Leave'})

# Eliminar filas con valores faltantes en las columnas especificadas
columns_to_check = ['ExperienceInCurrentDomain', 'JoiningYear']
data.dropna(subset=columns_to_check, inplace=True)

# Imputar datos faltantes en la columna 'Age' con la media
mean_age = data['Age'].mean()
data['Age'].fillna(mean_age, inplace=True)

# Imputar datos faltantes en la columna 'PaymentTier' con la moda
mode_payment_tier = data['PaymentTier'].mode().values[0]
data['PaymentTier'].fillna(mode_payment_tier, inplace=True)

# Eliminar registros con valores atípicos en columnas numéricas usando IQR
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
# Asegúrate de que la columna 'LeaveOrNot' no esté en la lista de columnas numéricas
if 'LeaveOrNot' in numeric_columns:
    numeric_columns.remove('LeaveOrNot')

Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Filtrar los registros que no tienen valores atípicos
filtered_data = data[~((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Mostrar el DataFrame actualizado
print("\nDataFrame actualizado después de eliminar registros con valores atípicos:")
print(filtered_data.head())

# Graficar la distribución de niveles de estudio usando subplots
education_counts = filtered_data['Education'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Histograma a la izquierda
axes[0].bar(education_counts.index, education_counts.values, color='skyblue')
axes[0].set_title('Distribución de Niveles de Estudio (Histograma)')
axes[0].set_xlabel('Nivel de Estudio')
axes[0].set_ylabel('Frecuencia')

# Gráfica de torta a la derecha
axes[1].pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axes[1].set_title('Distribución de Niveles de Estudio (Gráfica de Torta)')

plt.tight_layout()
plt.show()
