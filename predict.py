"""
Script de Predicción de Emociones
==================================
Proyecto Final - Inteligencia Artificial 2025-2
Universidad EAFIT
Autores: Miguel Villegas y Esteban Molina

Este script carga el modelo entrenado y permite clasificar emociones en textos en inglés.

Uso:
    # Modo interactivo
    python predict.py

    # Predecir un texto directo
    python predict.py "I am so happy today!"

    # Predecir desde archivo
    python predict.py --file texts.txt
"""

import joblib
import sys
import os
import numpy as np

class EmotionPredictor:
    """Clase para cargar y usar el modelo de clasificación de emociones"""

    def __init__(self, model_path='best_emotion_model.pkl'):
        """
        Inicializa el predictor cargando el modelo entrenado

        Args:
            model_path (str): Ruta al archivo del modelo guardado
        """
        self.model_path = model_path
        self.model_data = None
        self.vectorizer = None
        self.classifier = None
        self.emotion_names = None
        self.metrics = None

        self.load_model()

    def load_model(self):
        """Carga el modelo desde el archivo"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"\n Error: No se encontró el modelo en '{self.model_path}'\n"
                    f"   Por favor, ejecuta primero 'python main.py' para entrenar el modelo.\n"
                )

            print(f"\n{'='*80}")
            print("CARGANDO MODELO DE CLASIFICACIÓN DE EMOCIONES")
            print(f"{'='*80}\n")

            self.model_data = joblib.load(self.model_path)

            # Soporte para modelos con feature_union (ensemble) o vectorizer simple
            if 'feature_union' in self.model_data:
                self.vectorizer = self.model_data['feature_union']
                self.is_ensemble = True
            else:
                self.vectorizer = self.model_data['vectorizer']
                self.is_ensemble = False

            self.classifier = self.model_data['classifier']
            self.emotion_names = self.model_data['emotion_names']
            self.metrics = self.model_data['metrics']

            print(f" Modelo cargado exitosamente!")
            print(f"\n Información del modelo:")
            print(f"   • Nombre: {self.model_data['model_name']}")
            if self.is_ensemble:
                print(f"   • Tipo: Ensemble (Feature Engineering + Voting Classifier)")
            print(f"   • Accuracy: {self.metrics['accuracy']:.4f}")
            print(f"   • Precision (Macro): {self.metrics['precision_macro']:.4f}")
            print(f"   • Recall (Macro): {self.metrics['recall_macro']:.4f}")
            print(f"   • F1-Score (Macro): {self.metrics['f1_macro']:.4f}")
            print(f"\n Emociones detectables: {', '.join(self.emotion_names.values())}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n Error al cargar el modelo: {e}")
            sys.exit(1)

    def predict(self, text):
        """
        Predice la emoción de un texto

        Args:
            text (str): Texto en inglés a clasificar

        Returns:
            dict: Diccionario con la predicción y probabilidades
        """
        if not text or not text.strip():
            return {
                'error': 'El texto está vacío',
                'emotion': None,
                'confidence': None
            }

        try:
            # Vectorizar el texto
            text_vectorized = self.vectorizer.transform([text])

            # Hacer predicción
            prediction = self.classifier.predict(text_vectorized)[0]
            emotion = self.emotion_names[prediction]

            # Obtener probabilidades si el modelo lo soporta
            probabilities = None
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba(text_vectorized)[0]
                probabilities = {
                    self.emotion_names[i]: float(prob)
                    for i, prob in enumerate(probs)
                }
                confidence = float(probs[prediction])
            else:
                # Para modelos que no tienen predict_proba (como algunos Naive Bayes)
                confidence = None

            return {
                'text': text,
                'emotion': emotion,
                'emotion_id': int(prediction),
                'confidence': confidence,
                'probabilities': probabilities
            }

        except Exception as e:
            return {
                'error': f'Error al procesar el texto: {e}',
                'emotion': None,
                'confidence': None
            }

    def predict_batch(self, texts):
        """
        Predice emociones para múltiples textos

        Args:
            texts (list): Lista de textos en inglés

        Returns:
            list: Lista de diccionarios con predicciones
        """
        return [self.predict(text) for text in texts]

    def print_prediction(self, result):
        """
        Imprime el resultado de una predicción de forma formateada

        Args:
            result (dict): Resultado de la predicción
        """
        if 'error' in result and result['error']:
            print(f"\n {result['error']}")
            return

        print(f"\n{''*80}")
        print(f" Texto: \"{result['text'][:100]}{'...' if len(result['text']) > 100 else ''}\"")
        print(f"{''*80}")
        print(f" Emoción detectada: {result['emotion'].upper()}")

        if result['confidence'] is not None:
            print(f" Confianza: {result['confidence']*100:.2f}%")

            if result['probabilities']:
                print(f"\n Probabilidades por emoción:")
                # Ordenar por probabilidad descendente
                sorted_probs = sorted(
                    result['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for emotion, prob in sorted_probs:
                    bar_length = int(prob * 40)
                    bar = '' * bar_length + '' * (40 - bar_length)
                    marker = '←' if emotion == result['emotion'] else ''
                    print(f"   {emotion:12s} [{bar}] {prob*100:5.2f}% {marker}")

        print(f"{''*80}\n")


def interactive_mode(predictor):
    """Modo interactivo para ingresar múltiples textos"""
    print("\n MODO INTERACTIVO DE PREDICCIÓN DE EMOCIONES")
    print("="*80)
    print("Ingresa textos en inglés para clasificar sus emociones.")
    print("Comandos especiales:")
    print("  • 'exit' o 'quit' - Salir del programa")
    print("  • 'examples' - Ver ejemplos de textos")
    print("  • 'info' - Mostrar información del modelo")
    print("="*80 + "\n")

    while True:
        try:
            text = input("\n Ingresa un texto (o 'exit' para salir): ").strip()

            if not text:
                print("  Por favor ingresa un texto válido")
                continue

            if text.lower() in ['exit', 'quit', 'salir', 'q']:
                print("\n ¡Hasta luego!\n")
                break

            if text.lower() == 'examples':
                show_examples(predictor)
                continue

            if text.lower() == 'info':
                show_model_info(predictor)
                continue

            # Hacer predicción
            result = predictor.predict(text)
            predictor.print_prediction(result)

        except KeyboardInterrupt:
            print("\n\n Programa interrumpido. ¡Hasta luego!\n")
            break
        except Exception as e:
            print(f"\n Error inesperado: {e}\n")


def show_examples(predictor):
    """Muestra ejemplos de textos clasificados"""
    examples = {
        "Joy": [
            "I am so happy today!",
            "This is the best day of my life!",
            "I love spending time with my friends and family."
        ],
        "Sadness": [
            "I feel so lonely and sad.",
            "Everything seems dark and hopeless.",
            "I miss you so much."
        ],
        "Anger": [
            "I am so angry right now!",
            "This is completely unacceptable!",
            "How could you do this to me?"
        ],
        "Fear": [
            "I'm scared of what might happen.",
            "This is terrifying!",
            "I don't know if I can do this."
        ],
        "Love": [
            "I love you with all my heart.",
            "You mean everything to me.",
            "Being with you makes me complete."
        ],
        "Surprise": [
            "I can't believe this is happening!",
            "Wow, I never expected that!",
            "This is absolutely amazing!"
        ]
    }

    print(f"\n{'='*80}")
    print(" EJEMPLOS DE TEXTOS POR EMOCIÓN")
    print(f"{'='*80}\n")

    for emotion, texts in examples.items():
        print(f" {emotion}:")
        for i, text in enumerate(texts, 1):
            print(f"   {i}. \"{text}\"")
            result = predictor.predict(text)
            correct = "" if result['emotion'] == emotion else ""
            print(f"      → Predicción: {result['emotion']} {correct}")
        print()


def show_model_info(predictor):
    """Muestra información detallada del modelo"""
    print(f"\n{'='*80}")
    print(" INFORMACIÓN DETALLADA DEL MODELO")
    print(f"{'='*80}")
    print(f"\n  Nombre: {predictor.model_data['model_name']}")
    print(f"\n Métricas de rendimiento:")
    print(f"   • Accuracy: {predictor.metrics['accuracy']:.4f} ({predictor.metrics['accuracy']*100:.2f}%)")
    print(f"   • Precision (Macro): {predictor.metrics['precision_macro']:.4f} ({predictor.metrics['precision_macro']*100:.2f}%)")
    print(f"   • Recall (Macro): {predictor.metrics['recall_macro']:.4f} ({predictor.metrics['recall_macro']*100:.2f}%)")
    print(f"   • F1-Score (Macro): {predictor.metrics['f1_macro']:.4f} ({predictor.metrics['f1_macro']*100:.2f}%)")
    print(f"\n Emociones detectables:")
    for emotion_id, emotion_name in predictor.emotion_names.items():
        print(f"   {emotion_id}: {emotion_name}")
    print(f"{'='*80}\n")


def predict_from_file(predictor, file_path):
    """Predice emociones para textos desde un archivo"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"\n Procesando {len(texts)} textos desde '{file_path}'...\n")

        for i, text in enumerate(texts, 1):
            print(f"\n[{i}/{len(texts)}]")
            result = predictor.predict(text)
            predictor.print_prediction(result)

        print(f"\n Procesamiento completado: {len(texts)} textos clasificados.\n")

    except FileNotFoundError:
        print(f"\n Error: No se encontró el archivo '{file_path}'\n")
    except Exception as e:
        print(f"\n Error al leer el archivo: {e}\n")


def main():
    """Función principal"""
    # Crear predictor
    predictor = EmotionPredictor()

    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        # Modo de línea de comandos
        if sys.argv[1] == '--file' and len(sys.argv) > 2:
            # Predecir desde archivo
            predict_from_file(predictor, sys.argv[2])
        elif sys.argv[1] == '--examples':
            # Mostrar ejemplos
            show_examples(predictor)
        elif sys.argv[1] == '--info':
            # Mostrar información del modelo
            show_model_info(predictor)
        elif sys.argv[1] in ['-h', '--help']:
            # Mostrar ayuda
            print(__doc__)
        else:
            # Predecir un solo texto
            text = ' '.join(sys.argv[1:])
            result = predictor.predict(text)
            predictor.print_prediction(result)
    else:
        # Modo interactivo
        interactive_mode(predictor)


if __name__ == '__main__':
    main()
