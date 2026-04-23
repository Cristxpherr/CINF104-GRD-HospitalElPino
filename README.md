# CINF104-GRD-HospitalElPino
# Predicción de GRD - Hospital El Pino

Este repositorio contiene el código fuente y los recursos para la Fase 1 del proyecto de la asignatura **CINF104 Aprendizaje de Máquinas** de la Universidad Andrés Bello (UNAB).

## Descripción del Proyecto
El objetivo principal es construir un modelo de Machine Learning capaz de clasificar y predecir el código GRD (Grupos Relacionados por el Diagnóstico) de pacientes del Hospital El Pino, utilizando variables clínicas estructuradas (diagnósticos CIE-10, procedimientos CIE-9, edad y sexo).

## Contenido del Repositorio
* `grd_pipeline_complete.py`: Script principal de Python que ejecuta el pipeline completo. Incluye la carga de datos, el Análisis Exploratorio de Datos (EDA), el preprocesamiento (Multi-hot encoding mediante TF-IDF) y el entrenamiento/evaluación de los 4 modelos (Logistic Regression, Random Forest, XGBoost, MLP).
* `README.md`: Este archivo de documentación.

## Requisitos de Instalación
Para ejecutar el código localmente, asegúrate de tener instaladas las siguientes librerías:
\`\`\`bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
\`\`\`

## Ejecución
Para reproducir los resultados descritos en el paper, simplemente ejecuta el script principal asegurándote de actualizar la ruta del dataset dentro del archivo:
\`\`\`bash
python grd_pipeline_complete.py
\`\`\`

## Modelo Final Seleccionado
El modelo que presentó el mejor rendimiento en el conjunto de validación fue el **Multilayer Perceptron (MLP)**, logrando el Macro F1-score más alto al manejar adecuadamente la matriz dispersa de alta dimensionalidad.
