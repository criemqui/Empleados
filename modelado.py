import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Separar la columna objetivo
target = filtered_data['LeaveOrNot']
filtered_data = filtered_data.drop(columns=['LeaveOrNot'])

# Convertir las variables categóricas a variables dummies
filtered_data = pd.get_dummies(filtered_data)

# Mostrar el DataFrame actualizado sin la columna objetivo y con variables dummies
print("\nDataFrame actualizado después de convertir variables categóricas a dummies:")
print(filtered_data.head())

# Realizar la partición estratificada del dataset
X_train, X_test, y_train, y_test = train_test_split(filtered_data, target, test_size=0.2, random_state=42, stratify=target)

# Verificar la distribución de clases en los conjuntos de entrenamiento y prueba
print("\nDistribución de clases en el conjunto de entrenamiento:")
print(y_train.value_counts(normalize=True))
print("\nDistribución de clases en el conjunto de prueba:")
print(y_test.value_counts(normalize=True))

# Entrenar Random Forest sin cambios
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

# Entrenar Random Forest con class_weight="balanced"
rf_balanced = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_balanced.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred_default = rf_default.predict(X_test)
y_pred_balanced = rf_balanced.predict(X_test)

# Calcular la exactitud y mostrar el informe de clasificación
accuracy_default = accuracy_score(y_test, y_pred_default)
accuracy_balanced = accuracy_score(y_test, y_pred_balanced)

print("\nRandom Forest sin cambios:")
print(f"Exactitud: {accuracy_default:.4f}")
print(classification_report(y_test, y_pred_default))

print("\nRandom Forest con class_weight='balanced':")
print(f"Exactitud: {accuracy_balanced:.4f}")
print(classification_report(y_test, y_pred_balanced))
