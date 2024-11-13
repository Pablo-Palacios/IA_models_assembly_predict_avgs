# API de Cálculo de Promedio de Bateo

Esta es una API desarrollada en Flask que permite realizar cálculos y predicciones sobre el promedio de bateo utilizando varios modelos de machine learning. La API está diseñada para recibir datos de entrenamiento y realizar predicciones con diferentes modelos, además de brindar información sobre el rendimiento de estos modelos.

## Características

La API ofrece las siguientes funcionalidades:

1. **Predicción de Promedio de Bateo**:
   - Permite predecir el promedio de bateo usando tres modelos diferentes:
     - **Gradient Boosting**: Utiliza un modelo de regresión con Gradient Boosting.
     - **Random Forest**: Emplea un modelo de Random Forest para predicciones.
     - **XGBoost Regressor**: Realiza predicciones usando XGBoost para regresión.

2. **Entrenamiento de Modelos**:
   - Funcionalidad para entrenar modelos con un conjunto de datos específico (`data_set_mlb.csv`).
   - Almacena información del entrenamiento, como el error cuadrático medio (MSE) y los parámetros ajustados de cada modelo.

3. **Registro de Logs**:
   - Los eventos y errores del entrenamiento de modelos se registran en un archivo de log (`app/logs/training_models.log`), permitiendo un seguimiento detallado de la actividad de la API.
   - Al finalizar la ejecución, se almacenan detalles importantes sobre los datos de entrenamiento y predicción.




## Métricas de Evaluación

Durante el entrenamiento de los modelos, la API calcula varias métricas para evaluar el rendimiento de cada modelo. A continuación se describen las métricas de evaluación que se registran:

- **Score (score_gradient)**: Mide la precisión del modelo en el conjunto de entrenamiento. Representa el coeficiente de determinación \( R^2 \), que indica qué tan bien el modelo generaliza los datos. Un valor de 1 indica un modelo perfecto, mientras que un valor de 0 indica que el modelo no mejora la predicción promedio.

- **Error Cuadrático Medio (Mean Squared Error - MSE)**: Calculado con `mean_squared_gradient`, esta métrica mide el promedio de los cuadrados de los errores, es decir, la diferencia entre los valores predichos y los valores reales en el conjunto de prueba. Cuanto menor sea el valor del MSE, mejor es el rendimiento del modelo.

- **Error Absoluto Medio (Mean Absolute Error - MAE)**: Representado como `mean_absolute_gradient`, el MAE calcula el promedio de los errores absolutos, lo que significa la magnitud promedio de los errores sin considerar la dirección. Esta métrica es útil para interpretar el error medio en las mismas unidades que los datos de entrada.

- **Coeficiente de Determinación \( R^2 \) (R2 Score)**: `r2_score_gradient` representa la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes. Un valor cercano a 1 indica que el modelo explica la mayor parte de la variabilidad de los datos de salida.

- **Raíz del Error Cuadrático Medio (Root Mean Squared Error - RMSE)**: `root_mean_squared_gradient` es la raíz cuadrada del MSE y proporciona una interpretación en las mismas unidades que los datos originales. El RMSE es útil para tener una perspectiva directa de la desviación estándar del error de predicción.

### Ejemplo de Resultados de Métricas

Al finalizar el entrenamiento de un modelo, el archivo de log (`training_models.log`) almacena todas estas métricas para su revisión. Estos valores permiten comparar el rendimiento de diferentes modelos y ver cuál tiene mejor ajuste para el conjunto de datos utilizado.

