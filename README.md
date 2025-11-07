# Clasificación de Emociones en Texto (NLP Clásico)

Proyecto final del curso Inteligencia Artificial 2025-2 (Universidad EAFIT). El sistema clasifica la emoción predominante en textos cortos en inglés usando técnicas clásicas de NLP: representaciones BoW y TF‑IDF combinadas con modelos lineales (Naive Bayes, Regresión Logística y SGD), balanceo de clases con SMOTE y un ensemble con ingeniería de características. Incluye: pipeline de entrenamiento reproducible, visualizaciones, script CLI para inferencia y una aplicación web en Flask con interfaz estilo chat.

## Objetivos
- O1: Implementar modelos con TF‑IDF y BoW combinados con Naive Bayes y Regresión Logística.
- O2: Optimizar y evaluar para alcanzar ≥ 80% de precision macro en test, reportando recall, F1 por clase y matrices de confusión.

Resultado: el mejor modelo alcanzó Precision (macro) 85.79% y F1 (macro) 88.29% en el conjunto de prueba.

## Alcance y datos
- Idioma de entrada: inglés (se usan `stop_words='english'`).
- Emociones (multiclase, 6 clases): Sadness, Joy, Love, Anger, Fear, Surprise.
- Dataset: archivo `text.csv` con columnas `text` y `label` (0–5). El mapeo a nombres se define en el código de entrenamiento.

## Arquitectura del sistema
- Representación del texto: BoW y TF‑IDF (unigramas y bigramas, hasta 5000 features, `min_df=2`, `sublinear_tf=True` para TF‑IDF).
- Modelos baseline: MultinomialNB y LogisticRegression (multinomial, `lbfgs`).
- Optimización rápida: uso de hiperparámetros previamente validados para reducir tiempos (la metodología completa con GridSearchCV está documentada en OPTIMIZACION_IMPLEMENTADA.md).
- Balanceo: SMOTE dirigido a clases minoritarias (principalmente Love y Surprise) y `class_weight='balanced'` para LR.
- Ensemble: combinación ponderada de NB + LR + SGD sobre una unión de TF‑IDF/BoW con 18 features adicionales (longitud, mayúsculas, signos, intensificadores y listas de palabras emocionales) implementadas en `feature_engineering.py`.
- Persistencia: guardado del mejor modelo con `joblib` en `best_emotion_model.pkl` (incluye vectorizador/feature union, clasificador, métricas y nombres de clases).

## Métricas principales (mejor modelo)
- Accuracy: 91.13%
- Precision (Macro): 85.79%
- Recall (Macro): 91.95%
- F1‑Score (Macro): 88.29%

Notas de comportamiento:
- Buen desempeño general; mejora de recall en clases minoritarias tras SMOTE.
- Confusiones residuales típicas entre Fear/Anger y Joy/Love en algunos textos.

## Visualizaciones generadas
- `01_emotion_distribution.png`: distribución de clases.
- `02_text_length_analysis.png`: análisis de longitud del texto.
- `03_wordclouds.png`: nubes de palabras por emoción (opcional, requiere wordcloud).
- `04_model_comparison_baseline_vs_optimized.png`: comparación baseline vs optimizados.
- `04b_optimization_improvement.png`: mejoras porcentuales por optimización.
- `05_confusion_matrices.png`: matrices de confusión normalizadas.
- `06_best_model_metrics_by_class.png`: métricas por clase del mejor modelo.

## Estructura del repositorio (resumen)
- `main.py`: entrenamiento, evaluación, visualizaciones y guardado del mejor modelo.
- `feature_engineering.py`: extractor de 18 features adicionales para el ensemble.
- `predict.py`: CLI para cargar `best_emotion_model.pkl` y predecir emociones.
- `app.py`: servicio Flask para inferencia web (plantilla en `templates/index.html`).
- `requirements.txt`: dependencias del proyecto.
- Documentos de apoyo: `README_WEBAPP.md`, `INSTRUCCIONES_WEB.md`, `OPTIMIZACION_IMPLEMENTADA.md`, `VERIFICACION_MODELO.md`.

## Requisitos e instalación
1) Crear y activar un entorno virtual (opcional pero recomendado):
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
2) Instalar dependencias:
```
pip install -r requirements.txt
```
Opcional: para generar nubes de palabras, instala `wordcloud` (puede requerir cabeceras de compilación del sistema).

## Entrenamiento
Ejecuta el pipeline y genera artefactos (gráficas y modelo):
```
python main.py
```
Salida esperada:
- Gráficas PNG listadas en la sección “Visualizaciones”.
- Archivo `best_emotion_model.pkl` con el mejor modelo y sus componentes.

Notas sobre optimización:
- El script usa una “versión rápida” con hiperparámetros ya validados para reducir el tiempo de cómputo.
- La metodología completa de búsqueda con GridSearchCV está documentada y puede reactivarse si se desea replicar la búsqueda exhaustiva.

## Inferencia por línea de comandos (CLI)
Uso básico interactivo:
```
python predict.py
```
Predecir un texto directo:
```
python predict.py "I am so happy today!"
```
Predecir desde archivo (un texto por línea):
```
python predict.py --file texts.txt
```

## Aplicación web (Flask)
Inicia el servidor local:
```
python app.py
```
Luego abre en el navegador:
```
http://localhost:5000
```

Sin embargo, la página de flask, la cual usa el modelo, se encuentra desplegada, con Render, en el siguiente *enlace*:
```
https://emotion-classifier-9pee.onrender.com/
```

Características:
- Interfaz en una sola página estilo chat.
- Predicción con confidencias, distribución por emoción y palabras más influyentes.
- Modo comparación de dos textos, historial local y diseño responsive.

## Verificación rápida del modelo
Comprueba el archivo guardado y predicciones de ejemplo:
```
python test_model.py
```

## Limitaciones y consideraciones
- Entradas esperadas en inglés; para otros idiomas se requiere reentrenar y ajustar stopwords.
- El extractor de features adicionales usa listas de palabras emocionales curadas; puede sesgarse con jerga no contemplada.
- El sistema no realiza desambiguación semántica profunda; textos irónicos, figurativos o muy cortos pueden ser ambiguos.

## Autores
- Miguel Villegas
- Esteban Molina

Inteligencia Artificial 2025‑2

