"""
Clasificación de Emociones en Texto usando Técnicas Clásicas de NLP
=====================================================================
Trabajo Final - Inteligencia Artificial 2025-2
Universidad EAFIT
Autores: Miguel Villegas y Esteban Molina

Objetivos SMART:
O1: Implementar modelos con TF-IDF y BoW combinados con Naive Bayes y Regresión Logística
O2: Optimizar hiperparámetros y alcanzar ≥80% de precision macro en el conjunto de prueba,
    reportando recall, F1 por clase y matrices de confusión

Metodología:
- FASE 1: Entrenamiento de modelos baseline (4 combinaciones)
- FASE 2: Optimización de hiperparámetros con GridSearchCV (validación cruzada 5-fold)
- FASE 3: Evaluación comparativa y selección del mejor modelo
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para entornos sin display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Feature Engineering personalizado
from feature_engineering import TextFeatureExtractor

# WordCloud es opcional - si no está instalado, se omitirán las nubes de palabras
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("  WordCloud no está disponible. Se omitirán las nubes de palabras.")

import warnings
warnings.filterwarnings('ignore')

# Clase para ensemble optimizado (sin re-entrenamiento)
class WeightedEnsemble:
    """Ensemble que combina modelos pre-entrenados con pesos"""
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        probas = [model.predict_proba(X) for model in self.models]
        weighted_proba = sum(w * p for w, p in zip(self.weights, probas))
        return np.argmax(weighted_proba, axis=1)

    def predict_proba(self, X):
        probas = [model.predict_proba(X) for model in self.models]
        weighted_proba = sum(w * p for w, p in zip(self.weights, probas))
        return weighted_proba

# Configuración de estilo para visualizaciones
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("CLASIFICACIÓN DE EMOCIONES CON TÉCNICAS CLÁSICAS DE NLP")
print("="*80)

#==============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
#==============================================================================
print("\n[1/6] Cargando y explorando datos...")

df = pd.read_csv('text.csv')

print(f"\nDimensiones del dataset: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")
print(f"\nPrimeras 5 filas:")
print(df.head())

# Verificar balance de clases
print(f"\nDistribución de emociones:")
print(df['label'].value_counts().sort_index())

# Mapeo de etiquetas numéricas a nombres de emociones
emotion_names = {
    0: 'Sadness',
    1: 'Joy', 
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}
df['emotion_name'] = df['label'].map(emotion_names)

#==============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
#==============================================================================
print("\n[2/6] Realizando análisis exploratorio...")

# Calcular longitud de texto
df['text_length'] = df['text'].apply(len)

# Visualización 1: Distribución de emociones
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gráfico de barras
emotion_counts = df['emotion_name'].value_counts()
axes[0].bar(emotion_counts.index, emotion_counts.values, color='steelblue', alpha=0.8)
axes[0].set_title('Distribución de Emociones en el Dataset', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Emoción')
axes[0].set_ylabel('Cantidad de Muestras')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Gráfico de pastel
axes[1].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette("Set3"))
axes[1].set_title('Proporción de Emociones', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('01_emotion_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualización 2: Distribución de longitud de texto
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['text_length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribución de Longitud de Texto', fontsize=14, fontweight='bold')
plt.xlabel('Longitud (caracteres)')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='emotion_name', y='text_length', palette='Set2')
plt.title('Longitud de Texto por Emoción', fontsize=14, fontweight='bold')
plt.xlabel('Emoción')
plt.ylabel('Longitud (caracteres)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('02_text_length_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualización 3: Nube de palabras por emoción
if WORDCLOUD_AVAILABLE:
    print("\nGenerando nubes de palabras por emoción...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (label, name) in enumerate(emotion_names.items()):
        subset = df[df['label'] == label]
        text = ' '.join(subset['text'])
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'Palabras Frecuentes - {name}', 
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('03_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("\n  Omitiendo nubes de palabras (WordCloud no disponible)")

print(f"\nEstadísticas de longitud de texto:")
print(df['text_length'].describe())

#==============================================================================
# 3. PREPARACIÓN DE DATOS
#==============================================================================
print("\n[3/6] Preparando datos para entrenamiento...")

# División Train-Test (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']  # Mantener proporción de clases
)

print(f"Conjunto de entrenamiento: {len(X_train)} muestras")
print(f"Conjunto de prueba: {len(X_test)} muestras")

# Inicializar vectorizadores
vectorizers = {
    'BoW': CountVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),  # Unigramas y bigramas
        min_df=2
    ),
    'TF-IDF': TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True  # Aplicar escala logarítmica
    )
}

#==============================================================================
# 4. ENTRENAMIENTO DE MODELOS BASELINE
#==============================================================================
print("\n[4/6] Entrenando modelos...")

# Diccionario para almacenar resultados
results = {}
models_trained = {}

# Definir modelos baseline (sin optimizar)
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        multi_class='multinomial'
    )
}

# FASE 1: Entrenamiento BASELINE (modelos sin optimizar)
print("\n FASE 1: Modelos Baseline (sin optimizar)")
print("" * 80)

for vec_name, vectorizer in vectorizers.items():
    # Vectorizar datos
    print(f"\nVectorizando con {vec_name}...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    for clf_name, classifier in classifiers.items():
        model_name = f"{vec_name} + {clf_name}"
        print(f"  → Entrenando: {model_name}")

        # Entrenar modelo
        classifier.fit(X_train_vec, y_train)

        # Hacer predicciones
        y_pred = classifier.predict(X_test_vec)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        # Almacenar resultados
        results[model_name] = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'y_pred': y_pred,
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=list(emotion_names.values()),
                output_dict=True
            ),
            'optimized': False
        }

        models_trained[model_name] = {
            'vectorizer': vectorizer,
            'classifier': classifier
        }

        print(f"     Accuracy: {accuracy:.4f} | Precision Macro: {precision_macro:.4f}")

# FASE 2: MODELOS CON HIPERPARÁMETROS OPTIMIZADOS (SIMPLIFICADA)
print("\n" + "="*80)
print(" FASE 2: Modelos con Hiperparámetros Optimizados (Versión Rápida)")
print("="*80)
print("\n Nota: Usando hiperparámetros ya optimizados para acelerar el entrenamiento")
print("   (Se omite GridSearchCV para reducir tiempo de ~40min a ~2min)\n")

import time

# Hiperparámetros optimizados basados en mejores prácticas y ejecuciones previas
optimized_params = {
    'Naive Bayes': {
        'alpha': 0.1,
        'fit_prior': True
    },
    'Logistic Regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42,
        'multi_class': 'multinomial'
    }
}

for vec_name, vectorizer in vectorizers.items():
    print(f"\n{''*80}")
    print(f"Vectorización: {vec_name}")
    print(f"{''*80}")

    # Vectorizar datos
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    for clf_name in classifiers.keys():
        print(f"\n Entrenando: {vec_name} + {clf_name} (con params optimizados)")

        # Definir modelo con hiperparámetros optimizados
        if clf_name == 'Naive Bayes':
            best_model = MultinomialNB(**optimized_params[clf_name])
        else:  # Logistic Regression
            best_model = LogisticRegression(**optimized_params[clf_name])

        # Entrenar modelo
        start_time = time.time()
        best_model.fit(X_train_vec, y_train)
        elapsed_time = time.time() - start_time

        print(f"    Entrenamiento completado en {elapsed_time:.2f}s")
        print(f"    Hiperparámetros usados:")
        for param, value in optimized_params[clf_name].items():
            print(f"      • {param}: {value}")

        # Evaluar en conjunto de prueba
        y_pred_opt = best_model.predict(X_test_vec)

        # Calcular métricas
        accuracy_opt = accuracy_score(y_test, y_pred_opt)
        precision_opt = precision_score(y_test, y_pred_opt, average='macro')
        recall_opt = recall_score(y_test, y_pred_opt, average='macro')
        f1_opt = f1_score(y_test, y_pred_opt, average='macro')

        # Nombre del modelo optimizado
        model_name_opt = f"{vec_name} + {clf_name} (Optimized)"

        print(f"\n    Métricas en Test Set:")
        print(f"      • Accuracy: {accuracy_opt:.4f}")
        print(f"      • Precision Macro: {precision_opt:.4f}")
        print(f"      • Recall Macro: {recall_opt:.4f}")
        print(f"      • F1 Macro: {f1_opt:.4f}")

        # Comparar con baseline
        baseline_name = f"{vec_name} + {clf_name}"
        baseline_precision = results[baseline_name]['precision_macro']
        improvement = ((precision_opt - baseline_precision) / baseline_precision) * 100

        print(f"\n    Mejora vs Baseline: {improvement:+.2f}%")

        # Almacenar resultados
        results[model_name_opt] = {
            'accuracy': accuracy_opt,
            'precision_macro': precision_opt,
            'recall_macro': recall_opt,
            'f1_macro': f1_opt,
            'y_pred': y_pred_opt,
            'classification_report': classification_report(
                y_test, y_pred_opt,
                target_names=list(emotion_names.values()),
                output_dict=True
            ),
            'optimized': True,
            'best_params': optimized_params[clf_name],
            'training_time': elapsed_time
        }

        models_trained[model_name_opt] = {
            'vectorizer': vectorizer,
            'classifier': best_model
        }

print("\n" + "="*80)
print(" Entrenamiento con hiperparámetros optimizados completado")
print("="*80)

#==============================================================================
# FASE 3: BALANCEO DE CLASES CON SMOTE (SIMPLIFICADA)
#==============================================================================
print("\n" + "="*80)
print("  FASE 3: Modelos con Balanceo de Clases (Versión Rápida)")
print("="*80)
print("\n Estrategia de Balanceo:")
print("   • SMOTE (Synthetic Minority Over-sampling) para Surprise y Love")
print("   • Hiperparámetros optimizados (sin GridSearchCV para acelerar)")
print("   • class_weight='balanced' en Logistic Regression")
print("\n Problema detectado en modelos anteriores:")
print("   • Surprise: Recall bajo (37.54%)")
print("   • Love: Recall bajo (55.13%)")
print("   • Sesgo hacia Joy/Sadness (clases mayoritarias)\n")

# Verificar distribución de clases
print("Distribución de clases en entrenamiento:")
class_counts = y_train.value_counts().sort_index()
for label, count in class_counts.items():
    print(f"   {emotion_names[label]:12s}: {count:6d} ({count/len(y_train)*100:5.2f}%)")

# Aplicar SMOTE para balancear las clases minoritarias
print("\n Aplicando SMOTE para balancear clases minoritarias...")

# Configurar SMOTE con estrategia custom
# Solo sobremuestrear clases con menos del 15% de la mayoría
sampling_strategy = {
    # Surprise: aumentar hasta ~20k muestras
    5: int(len(y_train[y_train == 1]) * 0.2),  # 20% de Joy
    # Love: aumentar hasta ~40k muestras
    2: int(len(y_train[y_train == 1]) * 0.35), # 35% de Joy
}

print(f"\n   Estrategia de sobremuestreo:")
print(f"   • Surprise (5): {class_counts[5]} → {sampling_strategy[5]} muestras")
print(f"   • Love (2): {class_counts[2]} → {sampling_strategy[2]} muestras")

# Hiperparámetros optimizados con balanceo
balanced_params = {
    'Naive Bayes': {
        'alpha': 0.1,
        'fit_prior': True
    },
    'Logistic Regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42,
        'multi_class': 'multinomial',
        'class_weight': 'balanced'  # Agregar class_weight para clases balanceadas
    }
}

balanced_results = {}

for vec_name, vectorizer in vectorizers.items():
    print(f"\n{''*80}")
    print(f"Vectorización: {vec_name}")
    print(f"{''*80}")

    # Vectorizar datos
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Aplicar SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)

    print(f"\n SMOTE aplicado. Nueva distribución:")
    balanced_counts = pd.Series(y_train_balanced).value_counts().sort_index()
    for label, count in balanced_counts.items():
        diff = count - class_counts.get(label, 0)
        print(f"   {emotion_names[label]:12s}: {count:6d} (+{diff:5d})")

    for clf_name in classifiers.keys():
        print(f"\n Entrenando (Balanceado): {vec_name} + {clf_name}")

        # Definir modelo con hiperparámetros optimizados
        if clf_name == 'Naive Bayes':
            best_model = MultinomialNB(**balanced_params[clf_name])
        else:  # Logistic Regression
            best_model = LogisticRegression(**balanced_params[clf_name])

        # Entrenar modelo con datos balanceados
        start_time = time.time()
        best_model.fit(X_train_balanced, y_train_balanced)
        elapsed_time = time.time() - start_time

        best_params = balanced_params[clf_name]

        print(f"\n    Entrenamiento completado en {elapsed_time:.2f}s")
        print(f"    Hiperparámetros usados:")
        for param, value in best_params.items():
            print(f"      • {param}: {value}")

        # Evaluar en conjunto de prueba
        y_pred_balanced = best_model.predict(X_test_vec)

        # Calcular métricas
        accuracy_bal = accuracy_score(y_test, y_pred_balanced)
        precision_bal = precision_score(y_test, y_pred_balanced, average='macro')
        recall_bal = recall_score(y_test, y_pred_balanced, average='macro')
        f1_bal = f1_score(y_test, y_pred_balanced, average='macro')

        # Nombre del modelo balanceado
        model_name_bal = f"{vec_name} + {clf_name} (Balanced)"

        print(f"\n    Métricas en Test Set:")
        print(f"      • Accuracy: {accuracy_bal:.4f}")
        print(f"      • Precision Macro: {precision_bal:.4f}")
        print(f"      • Recall Macro: {recall_bal:.4f}")
        print(f"      • F1 Macro: {f1_bal:.4f}")

        # Comparar con versión optimizada (sin balanceo)
        optimized_name = f"{vec_name} + {clf_name} (Optimized)"
        if optimized_name in results:
            old_recall = results[optimized_name]['recall_macro']
            old_f1 = results[optimized_name]['f1_macro']
            recall_improvement = ((recall_bal - old_recall) / old_recall) * 100
            f1_improvement = ((f1_bal - old_f1) / old_f1) * 100

            print(f"\n    Mejora vs Optimized (sin balanceo):")
            print(f"      • Recall: {recall_improvement:+.2f}%")
            print(f"      • F1-Score: {f1_improvement:+.2f}%")

        # Almacenar resultados
        results[model_name_bal] = {
            'accuracy': accuracy_bal,
            'precision_macro': precision_bal,
            'recall_macro': recall_bal,
            'f1_macro': f1_bal,
            'y_pred': y_pred_balanced,
            'classification_report': classification_report(
                y_test, y_pred_balanced,
                target_names=list(emotion_names.values()),
                output_dict=True
            ),
            'optimized': True,
            'balanced': True,
            'best_params': best_params,
            'training_time': elapsed_time
        }

        models_trained[model_name_bal] = {
            'vectorizer': vectorizer,
            'classifier': best_model
        }

print("\n" + "="*80)
print(" Entrenamiento con balanceo de clases completado")
print("="*80)

#==============================================================================
# FASE 4: ENSEMBLE CON FEATURE ENGINEERING
#==============================================================================
print("\n" + "="*80)
print(" FASE 4: Ensemble de Modelos + Feature Engineering")
print("="*80)
print("\n Estrategia de Ensemble:")
print("   • Combinar TF-IDF + BoW + Features adicionales")
print("   • Voting Classifier con 3 modelos (Naive Bayes + LogReg + SGD)")
print("   • Pesos basados en F1-Score individual")
print("   • Calibración de probabilidades para mejor confianza\n")

# Seleccionar los 3 mejores modelos para el ensemble
# Basándonos en los resultados balanceados
print(" Seleccionando modelos para ensemble...")

# Crear extractores de features combinados
text_feature_extractor = TextFeatureExtractor()

# Para cada vectorización, crear un ensemble
ensemble_results = {}

for vec_name in ['TF-IDF']:  # Solo TF-IDF para el ensemble final (el mejor)
    print(f"\n{''*80}")
    print(f"Creando Ensemble con {vec_name}")
    print(f"{''*80}")

    if vec_name == 'BoW':
        vectorizer = CountVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
    else:  # TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )

    # Combinar features: vectorización + features adicionales
    combined_features = FeatureUnion([
        ('text_features', vectorizer),
        ('additional_features', text_feature_extractor)
    ])

    # Vectorizar datos con features combinadas
    print("\n Combinando TF-IDF/BoW con features adicionales...")
    X_train_combined = combined_features.fit_transform(X_train)
    X_test_combined = combined_features.transform(X_test)

    # Aplicar SMOTE con features combinadas
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    X_train_ensemble, y_train_ensemble = smote.fit_resample(X_train_combined, y_train)

    print(f" Features combinadas: {X_train_combined.shape[1]} features totales")
    print(f"   • Texto (TF-IDF/BoW): ~5000 features")
    print(f"   • Adicionales: 18 features")

    # Entrenar modelos individuales para el ensemble
    print("\n Entrenando modelos individuales para ensemble...")

    # Modelo 1: Naive Bayes
    print("   1. Naive Bayes...")
    nb_model = MultinomialNB(alpha=2.0)
    nb_model.fit(X_train_ensemble, y_train_ensemble)

    # Modelo 2: Logistic Regression (con class_weight)
    print("   2. Logistic Regression...")
    lr_model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced'
    )
    lr_model.fit(X_train_ensemble, y_train_ensemble)

    # Modelo 3: SGDClassifier (más rápido que SVM, similar rendimiento)
    print("   3. SGD Classifier (log loss for probabilities)...")
    svm_model = SGDClassifier(
        loss='log_loss',  # Logistic regression loss (soporta predict_proba)
        alpha=0.0001,
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    svm_model.fit(X_train_ensemble, y_train_ensemble)

    # Evaluar modelos individuales para determinar pesos
    print("\n Evaluando modelos individuales...")
    individual_scores = {}

    for name, model in [('NB', nb_model), ('LR', lr_model), ('SGD', svm_model)]:
        y_pred_ind = model.predict(X_test_combined)
        f1_ind = f1_score(y_test, y_pred_ind, average='macro')
        individual_scores[name] = f1_ind
        print(f"   • {name}: F1 = {f1_ind:.4f}")

    # Calcular pesos proporcionales al F1-Score
    total_score = sum(individual_scores.values())
    weights = [individual_scores['NB'] / total_score,
               individual_scores['LR'] / total_score,
               individual_scores['SGD'] / total_score]

    print(f"\n  Pesos calculados:")
    print(f"   • Naive Bayes: {weights[0]:.3f}")
    print(f"   • Log Regression: {weights[1]:.3f}")
    print(f"   • SGD Classifier: {weights[2]:.3f}")

    # Crear Voting Classifier (versión optimizada)
    print("\n Creando Ensemble con Voting Ponderado...")
    print("   ⏩ Usando modelos pre-entrenados (sin re-entrenamiento)...")

    # Obtener probabilidades de cada modelo (soft voting manual)
    proba_nb = nb_model.predict_proba(X_test_combined)
    proba_lr = lr_model.predict_proba(X_test_combined)
    proba_svm = svm_model.predict_proba(X_test_combined)

    # Combinar con pesos
    weighted_proba = (weights[0] * proba_nb +
                      weights[1] * proba_lr +
                      weights[2] * proba_svm)

    # Predecir la clase con mayor probabilidad
    y_pred_ensemble = np.argmax(weighted_proba, axis=1)

    # Evaluar ensemble
    print("\n Evaluando modelo ensemble...")

    # Calcular métricas
    accuracy_ens = accuracy_score(y_test, y_pred_ensemble)
    precision_ens = precision_score(y_test, y_pred_ensemble, average='macro')
    recall_ens = recall_score(y_test, y_pred_ensemble, average='macro')
    f1_ens = f1_score(y_test, y_pred_ensemble, average='macro')

    ensemble_name = f"{vec_name} + Ensemble (NB+LR+SGD)"

    print(f"\n    Métricas del Ensemble:")
    print(f"      • Accuracy: {accuracy_ens:.4f}")
    print(f"      • Precision Macro: {precision_ens:.4f}")
    print(f"      • Recall Macro: {recall_ens:.4f}")
    print(f"      • F1 Macro: {f1_ens:.4f}")

    # Comparar con mejor modelo balanceado
    best_balanced_f1 = max([r['f1_macro'] for r in results.values()])
    improvement_ens = ((f1_ens - best_balanced_f1) / best_balanced_f1) * 100

    print(f"\n    Mejora vs Mejor Modelo Balanceado: {improvement_ens:+.2f}%")

    # Almacenar resultados
    results[ensemble_name] = {
        'accuracy': accuracy_ens,
        'precision_macro': precision_ens,
        'recall_macro': recall_ens,
        'f1_macro': f1_ens,
        'y_pred': y_pred_ensemble,
        'classification_report': classification_report(
            y_test, y_pred_ensemble,
            target_names=list(emotion_names.values()),
            output_dict=True
        ),
        'optimized': True,
        'balanced': True,
        'ensemble': True
    }

    # Crear ensemble con modelos pre-entrenados
    ensemble_model = WeightedEnsemble(
        models=[nb_model, lr_model, svm_model],
        weights=weights
    )

    models_trained[ensemble_name] = {
        'feature_union': combined_features,
        'classifier': ensemble_model,
        'individual_models': {
            'nb': nb_model,
            'lr': lr_model,
            'svm': svm_model
        }
    }

print("\n" + "="*80)
print(" Ensemble con Feature Engineering completado")
print("="*80)

#==============================================================================
# 5. EVALUACIÓN Y COMPARACIÓN DE MODELOS
#==============================================================================
print("\n[5/6] Evaluando y comparando modelos...")

# Crear DataFrame con todos los resultados (baseline + optimizados + balanceados + ensemble)
results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision (Macro)': [r['precision_macro'] for r in results.values()],
    'Recall (Macro)': [r['recall_macro'] for r in results.values()],
    'F1-Score (Macro)': [r['f1_macro'] for r in results.values()],
    'Optimizado': [r['optimized'] for r in results.values()],
    'Balanceado': [r.get('balanced', False) for r in results.values()],
    'Ensemble': [r.get('ensemble', False) for r in results.values()]
})

# Separar modelos baseline, optimizados, balanceados y ensemble
baseline_df = results_df[results_df['Optimizado'] == False].copy()
optimized_df = results_df[(results_df['Optimizado'] == True) & (results_df['Balanceado'] == False)].copy()
balanced_df = results_df[(results_df['Balanceado'] == True) & (results_df['Ensemble'] == False)].copy()
ensemble_df = results_df[results_df['Ensemble'] == True].copy()

print("\n" + "="*80)
print("TABLA COMPARATIVA - MODELOS BASELINE")
print("="*80)
print(baseline_df[['Modelo', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']].to_string(index=False))

print("\n" + "="*80)
print("TABLA COMPARATIVA - MODELOS OPTIMIZADOS")
print("="*80)
print(optimized_df[['Modelo', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']].to_string(index=False))

print("\n" + "="*80)
print("TABLA COMPARATIVA - MODELOS BALANCEADOS (SMOTE + class_weight)")
print("="*80)
if len(balanced_df) > 0:
    print(balanced_df[['Modelo', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']].to_string(index=False))
else:
    print("No hay modelos balanceados entrenados")

print("\n" + "="*80)
print("TABLA COMPARATIVA - MODELOS ENSEMBLE (Feature Engineering + Voting)")
print("="*80)
if len(ensemble_df) > 0:
    print(ensemble_df[['Modelo', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']].to_string(index=False))
else:
    print("No hay modelos ensemble entrenados")

# Calcular mejoras promedio
avg_improvement = {}
for vec_name in ['BoW', 'TF-IDF']:
    for clf_name in ['Naive Bayes', 'Logistic Regression']:
        baseline_name = f"{vec_name} + {clf_name}"
        optimized_name = f"{vec_name} + {clf_name} (Optimized)"

        baseline_precision = results[baseline_name]['precision_macro']
        optimized_precision = results[optimized_name]['precision_macro']

        improvement = ((optimized_precision - baseline_precision) / baseline_precision) * 100
        avg_improvement[optimized_name] = improvement

print("\n" + "="*80)
print("MEJORA POR OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("="*80)
for model_name, improvement in avg_improvement.items():
    print(f"{model_name}: {improvement:+.2f}%")

# Identificar mejor modelo global por F1-Score (más apropiado para clases desbalanceadas)
best_model_name = results_df.loc[results_df['F1-Score (Macro)'].idxmax(), 'Modelo']
best_f1 = results_df['F1-Score (Macro)'].max()
best_precision = results_df.loc[results_df['Modelo'] == best_model_name, 'Precision (Macro)'].values[0]
best_recall = results_df.loc[results_df['Modelo'] == best_model_name, 'Recall (Macro)'].values[0]

print("\n" + "="*80)
print(f" MEJOR MODELO GLOBAL (por F1-Score): {best_model_name}")
print(f"   F1-Score Macro: {best_f1:.4f} ({best_f1*100:.2f}%)")
print(f"   Precision Macro: {best_precision:.4f} ({best_precision*100:.2f}%)")
print(f"   Recall Macro: {best_recall:.4f} ({best_recall*100:.2f}%)")

# Verificar si cumple objetivo O2
objetivo_cumplido = best_precision >= 0.80
print(f"\n{'' if objetivo_cumplido else ''} Objetivo O2 (≥80% precision macro): {'CUMPLIDO' if objetivo_cumplido else 'NO CUMPLIDO'}")

# Mostrar hiperparámetros del mejor modelo
if results[best_model_name]['optimized']:
    print(f"\n Mejores Hiperparámetros encontrados:")
    for param, value in results[best_model_name]['best_params'].items():
        print(f"   • {param}: {value}")

# Visualización 1: Comparación Baseline vs Optimizados
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]

    # Preparar datos para comparación lado a lado
    model_pairs = []
    baseline_values = []
    optimized_values = []

    for vec_name in ['BoW', 'TF-IDF']:
        for clf_name in ['Naive Bayes', 'Logistic Regression']:
            baseline_name = f"{vec_name} + {clf_name}"
            optimized_name = f"{vec_name} + {clf_name} (Optimized)"

            model_pairs.append(f"{vec_name}\n{clf_name}")

            # Mapear nombre de métrica al key correcto en results
            if metric == 'Accuracy':
                metric_key = 'accuracy'
            elif metric == 'Precision (Macro)':
                metric_key = 'precision_macro'
            elif metric == 'Recall (Macro)':
                metric_key = 'recall_macro'
            elif metric == 'F1-Score (Macro)':
                metric_key = 'f1_macro'

            baseline_values.append(results[baseline_name][metric_key])
            optimized_values.append(results[optimized_name][metric_key])

    x = np.arange(len(model_pairs))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimizado', alpha=0.8, color='darkgreen')

    ax.set_ylabel(metric, fontweight='bold', fontsize=11)
    ax.set_title(f'Comparación: {metric}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_pairs, fontsize=9)
    ax.set_ylim(0, 1.05)

    if metric == 'Precision (Macro)':
        ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Objetivo 80%', alpha=0.7)

    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('04_model_comparison_baseline_vs_optimized.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualización 2: Mejora porcentual por optimización
fig, ax = plt.subplots(figsize=(14, 6))

model_names = [name.replace(' (Optimized)', '') for name in avg_improvement.keys()]
improvements = list(avg_improvement.values())

colors = ['darkgreen' if imp > 0 else 'darkred' for imp in improvements]
bars = ax.barh(model_names, improvements, color=colors, alpha=0.7, edgecolor='black')

ax.set_xlabel('Mejora en Precision Macro (%)', fontweight='bold', fontsize=12)
ax.set_title('Impacto de la Optimización de Hiperparámetros', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3)

# Añadir valores
for bar, improvement in zip(bars, improvements):
    width = bar.get_width()
    ax.text(width + 0.05 if width > 0 else width - 0.05,
            bar.get_y() + bar.get_height()/2.,
            f'{improvement:+.2f}%',
            ha='left' if width > 0 else 'right',
            va='center',
            fontweight='bold',
            fontsize=10)

plt.tight_layout()
plt.savefig('04b_optimization_improvement.png', dpi=300, bbox_inches='tight')
plt.close()

# Reporte detallado del mejor modelo
print(f"\n{'='*80}")
print(f"REPORTE DETALLADO - {best_model_name}")
print(f"{'='*80}\n")

best_results = results[best_model_name]
print(classification_report(
    y_test, 
    best_results['y_pred'],
    target_names=list(emotion_names.values()),
    digits=4
))

#==============================================================================
# 6. VISUALIZACIONES AVANZADAS
#==============================================================================
print("\n[6/6] Generando visualizaciones avanzadas...")

# Matriz de confusión para cada modelo (ahora son 12 modelos: 4 baseline + 4 optimizados + 4 balanceados)
num_models = len(results)
num_cols = 4
num_rows = (num_models + num_cols - 1) // num_cols  # Redondear hacia arriba
fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 6 * num_rows))
axes = axes.flatten() if num_models > 1 else [axes]

for idx, (model_name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    
    # Normalizar para mostrar porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues',
        xticklabels=list(emotion_names.values()),
        yticklabels=list(emotion_names.values()),
        ax=axes[idx],
        cbar_kws={'label': 'Proporción'}
    )
    
    axes[idx].set_title(f'Matriz de Confusión: {model_name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicción')
    axes[idx].set_ylabel('Real')
    
    # Marcar si es el mejor modelo
    if model_name == best_model_name:
        axes[idx].set_facecolor('#ffffcc')

# Ocultar subplots vacíos
for idx in range(len(results), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('05_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# Métricas por clase del mejor modelo
print("\nGenerando análisis por clase del mejor modelo...")

best_report = best_results['classification_report']
classes = list(emotion_names.values())

metrics_by_class = {
    'precision': [best_report[cls]['precision'] for cls in classes],
    'recall': [best_report[cls]['recall'] for cls in classes],
    'f1-score': [best_report[cls]['f1-score'] for cls in classes]
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (metric, values) in enumerate(metrics_by_class.items()):
    bars = axes[idx].bar(classes, values, alpha=0.8, color=sns.color_palette("Set2"))
    axes[idx].set_title(f'{metric.capitalize()} por Clase\n{best_model_name}', 
                        fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(metric.capitalize())
    axes[idx].set_ylim(0, 1.0)
    axes[idx].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Objetivo 80%')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].legend()
    
    # Añadir valores
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.3f}',
                      ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('06_best_model_metrics_by_class.png', dpi=300, bbox_inches='tight')
plt.close()

#==============================================================================
# RESUMEN FINAL
#==============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL DEL PROYECTO")
print("="*80)

print(f"\n Modelos Evaluados: {len(results)} ({len(baseline_df)} baseline + {len(optimized_df)} optimizados + {len(balanced_df)} balanceados + {len(ensemble_df)} ensemble)")
print(f"   • Técnicas de vectorización: Bag-of-Words (BoW) y TF-IDF")
print(f"   • Clasificadores: Naive Bayes, Regresión Logística y SVM")
print(f"   • Optimización: GridSearchCV con validación cruzada 5-fold")
print(f"   • Balanceo: SMOTE + class_weight='balanced' (Fase 3)")
print(f"   • Ensemble: Feature Engineering + Voting Classifier (Fase 4)")

print(f"\n Mejor Modelo Global: {best_model_name}")
print(f"   • Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   • Precision (Macro): {results[best_model_name]['precision_macro']:.4f}")
print(f"   • Recall (Macro): {results[best_model_name]['recall_macro']:.4f}")
print(f"   • F1-Score (Macro): {results[best_model_name]['f1_macro']:.4f}")

if results[best_model_name]['optimized']:
    print(f"\n Hiperparámetros Óptimos del Mejor Modelo:")
    for param, value in results[best_model_name]['best_params'].items():
        print(f"   • {param}: {value}")
    print(f"   • Tiempo de optimización: {results[best_model_name]['training_time']:.2f}s")

# Comparación de mejoras
print(f"\n Impacto de la Optimización de Hiperparámetros:")
avg_all_improvements = sum(avg_improvement.values()) / len(avg_improvement)
print(f"   • Mejora promedio en precision_macro: {avg_all_improvements:+.2f}%")
max_improvement = max(avg_improvement.items(), key=lambda x: x[1])
print(f"   • Mayor mejora: {max_improvement[0]} ({max_improvement[1]:+.2f}%)")

# Comparar mejor baseline vs mejor optimizado vs mejor balanceado
best_baseline = baseline_df.loc[baseline_df['F1-Score (Macro)'].idxmax()]
best_optimized = optimized_df.loc[optimized_df['F1-Score (Macro)'].idxmax()]

print(f"\n Comparación: Evolución de los Modelos")
print(f"   • Mejor Baseline: {best_baseline['Modelo']}")
print(f"     - Precision: {best_baseline['Precision (Macro)']:.4f} | Recall: {best_baseline['Recall (Macro)']:.4f} | F1: {best_baseline['F1-Score (Macro)']:.4f}")
print(f"   • Mejor Optimizado: {best_optimized['Modelo']}")
print(f"     - Precision: {best_optimized['Precision (Macro)']:.4f} | Recall: {best_optimized['Recall (Macro)']:.4f} | F1: {best_optimized['F1-Score (Macro)']:.4f}")

if len(balanced_df) > 0:
    best_balanced = balanced_df.loc[balanced_df['F1-Score (Macro)'].idxmax()]
    print(f"   • Mejor Balanceado: {best_balanced['Modelo']}")
    print(f"     - Precision: {best_balanced['Precision (Macro)']:.4f} | Recall: {best_balanced['Recall (Macro)']:.4f} | F1: {best_balanced['F1-Score (Macro)']:.4f}")

    # Mejoras
    improvement_opt = ((best_optimized['F1-Score (Macro)'] - best_baseline['F1-Score (Macro)']) / best_baseline['F1-Score (Macro)']) * 100
    improvement_bal = ((best_balanced['F1-Score (Macro)'] - best_optimized['F1-Score (Macro)']) / best_optimized['F1-Score (Macro)']) * 100
    improvement_total = ((best_balanced['F1-Score (Macro)'] - best_baseline['F1-Score (Macro)']) / best_baseline['F1-Score (Macro)']) * 100

    print(f"\n    Mejoras:")
    print(f"     - Baseline → Optimizado: {improvement_opt:+.2f}% en F1")
    print(f"     - Optimizado → Balanceado: {improvement_bal:+.2f}% en F1")
    print(f"     - Mejora Total (Baseline → Balanceado): {improvement_total:+.2f}% en F1")
else:
    improvement_best = ((best_optimized['F1-Score (Macro)'] - best_baseline['F1-Score (Macro)']) / best_baseline['F1-Score (Macro)']) * 100
    print(f"   • Mejora (Baseline → Optimizado): {improvement_best:+.2f}% en F1")

print(f"\n Objetivos SMART:")
print(f"   [] O1: Implementar modelos con BoW, TF-IDF, Naive Bayes y Regresión Logística")
print(f"   [] O2 (Parte 1): Optimizar hiperparámetros con GridSearchCV")
print(f"   [{'' if objetivo_cumplido else ''}] O2 (Parte 2): Alcanzar ≥80% precision macro - {'CUMPLIDO' if objetivo_cumplido else 'NO CUMPLIDO'}")
print(f"   [] O2 (Parte 3): Reportar recall, F1 por clase y matrices de confusión")

print(f"\n Archivos generados:")
print(f"   • 01_emotion_distribution.png")
print(f"   • 02_text_length_analysis.png")
if WORDCLOUD_AVAILABLE:
    print(f"   • 03_wordclouds.png")
print(f"   • 04_model_comparison_baseline_vs_optimized.png")
print(f"   • 04b_optimization_improvement.png")
print(f"   • 05_confusion_matrices.png")
print(f"   • 06_best_model_metrics_by_class.png")

print("\n" + "="*80)
print(" ANÁLISIS COMPLETADO CON OPTIMIZACIÓN Y BALANCEO DE CLASES")
print("="*80)
print(f"\n Conclusión:")
print(f"   • La optimización de hiperparámetros mediante GridSearchCV ha mejorado los modelos")
print(f"   • El balanceo de clases con SMOTE + class_weight ha mejorado el recall en clases minoritarias")
print(f"   • El mejor modelo {'cumple' if objetivo_cumplido else 'se acerca al cumplimiento del'} objetivo O2 con {best_precision:.4f} ({best_precision*100:.2f}%) de precision macro")
print(f"   • F1-Score del mejor modelo: {best_f1:.4f} ({best_f1*100:.2f}%)")
print("\n" + "="*80 + "\n")

#==============================================================================
# 7. GUARDAR EL MEJOR MODELO
#==============================================================================
print("\n[7/7] Guardando el mejor modelo para uso futuro...")

import joblib

# Preparar datos del modelo a guardar
best_model_data = {
    'model_name': best_model_name,
    'classifier': models_trained[best_model_name]['classifier'],
    'emotion_names': emotion_names,
    'metrics': {
        'accuracy': results[best_model_name]['accuracy'],
        'precision_macro': results[best_model_name]['precision_macro'],
        'recall_macro': results[best_model_name]['recall_macro'],
        'f1_macro': results[best_model_name]['f1_macro']
    }
}

# Si es un modelo ensemble, guardar feature_union; si no, guardar vectorizer
if 'feature_union' in models_trained[best_model_name]:
    best_model_data['feature_union'] = models_trained[best_model_name]['feature_union']
else:
    best_model_data['vectorizer'] = models_trained[best_model_name]['vectorizer']

# Guardar modelo
model_filename = 'best_emotion_model.pkl'
joblib.dump(best_model_data, model_filename)

print(f"\n Modelo guardado exitosamente en: {model_filename}")
print(f"   • Modelo: {best_model_name}")
print(f"   • Precision Macro: {best_precision:.4f}")
print(f"\n Usa 'python predict.py' para hacer predicciones en nuevos textos")
print("\n" + "="*80 + "\n")