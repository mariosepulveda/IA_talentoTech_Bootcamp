from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


#Cargar el conjunto de datos Iris
iris = load_iris()

# Convertir a dataFrame para manipulación y visualización
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()

#añadimos nombres de clases para mejorar comprensión visual
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
# graficar caracteristicas
sns.pairplot(df, hue='target', markers=["o","s","D"])
plt.show()

# configure el tamaño de la figura
plt.figure(figsize=(12,8))
# Grafica de caja para cada característica agrupada por especie
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(f'Boxplot de {feature}')
plt.tight_layout()
plt.show()

# configure el tamaño de la figura
plt.figure(figsize=(12,8))
# Grafica de violín para cada característica agrupada por especie
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='target', y=feature, data=df)
    plt.title(f'Violin plot de {feature}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Normalizar características
scaler = MinMaxScaler()
df[iris.feature_names] = scaler.fit_transform(df[iris.feature_names])
df.head()


#inicializar el modelo secuencial
model = Sequential()
#capa de entrada (4 características de entrada y la primnera capa oculta 8 neuronas )
model.add(Dense(8, input_dim=4, activation='relu'))
#capa oculta con 6 neuronas
model.add(Dense(6, activation='relu'))
#capa de salida con 3 neuronas
model.add(Dense(3, activation='softmax'))
#Resumen de la arquitectura de la red
model.summary()

# compilación de modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#cargar el conjuento de datos iris
iris = load_iris()
x = iris.data
y = iris.target
# convertir las etiquetas a codificacíon categórias
y_categorical = to_categorical(y)
# Dividir el conjunto de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y_categorical, test_size=0.2, random_state=42)
# Inicializar el modelo secuencial
model = Sequential()
#capa de entreda
model.add(Dense(8,input_shape=(4,),activation='relu'))
#capa oculta
model.add(Dense(6,activation='relu'))
#capa de salida
model.add(Dense(3,activation='softmax'))
# compilar el modelo con función de pérdida y optimizador
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# entrenar el modelo
model.fit(x_train, y_train, epochs=100, batch_size=4, validation_data=(x_test, y_test))
#Guardar como archivo HDF5
model.save('iris_model.h5')
print("Modelo guardado como iris_model.h5")