"""
Feature Engineering para Clasificación de Emociones
===================================================
Proyecto Final - Inteligencia Artificial 
Por: Miguel Villegas y Esteban Molina

Este módulo contiene extractores de features adicionales para mejorar
la clasificación de emociones en textos cortos y ambiguos.
"""

import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extractor de features adicionales de texto para mejorar la clasificación

    Features extraídas:
    - Longitud del texto (palabras y caracteres)
    - Conteo de signos de exclamación, interrogación, puntos suspensivos
    - Ratio de palabras en mayúsculas
    - Presencia de palabras emocionales clave
    - Densidad de adjetivos/adverbios intensificadores
    """

    def __init__(self):
        # Diccionarios de palabras clave por emoción
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'amazing', 'wonderful', 'great',
                   'fantastic', 'love', 'delighted', 'awesome', 'perfect', 'yay',
                   'haha', 'lol', 'smile', 'glad', 'pleased', 'cheerful'],
            'sadness': ['sad', 'unhappy', 'depressed', 'lonely', 'miserable',
                       'cry', 'tears', 'down', 'blue', 'upset', 'disappointed',
                       'hurt', 'pain', 'miss', 'lost', 'empty', 'hopeless'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed',
                     'irritated', 'frustrated', 'pissed', 'outraged', 'livid',
                     'disgusted', 'ugh', 'damn', 'stupid', 'idiot'],
            'fear': ['scared', 'afraid', 'terrified', 'fear', 'worried',
                    'anxious', 'nervous', 'panic', 'frightened', 'alarmed',
                    'dread', 'terror', 'phobia', 'stress', 'uneasy'],
            'love': ['love', 'adore', 'cherish', 'affection', 'dear', 'darling',
                    'sweetheart', 'beloved', 'heart', 'romantic', 'passion',
                    'forever', 'soulmate', 'treasure', 'devoted'],
            'surprise': ['surprise', 'shocked', 'amazed', 'astonished', 'wow',
                        'omg', 'unbelievable', 'incredible', 'unexpected',
                        'stunned', 'speechless', 'cannot believe', 'didnt expect']
        }

        # Intensificadores
        self.intensifiers = ['very', 'so', 'really', 'extremely', 'incredibly',
                           'absolutely', 'completely', 'totally', 'utterly',
                           'highly', 'quite', 'rather', 'pretty', 'super']

    def fit(self, X, y=None):
        """Fit no hace nada, pero es necesario para compatibilidad con sklearn"""
        return self

    def transform(self, X):
        """
        Transforma una lista de textos en una matriz de features

        Args:
            X: Lista o Series de textos

        Returns:
            numpy array con las features extraídas
        """
        features = []

        for text in X:
            text_lower = str(text).lower()
            words = text_lower.split()

            # 1. Longitud del texto
            num_chars = len(text)
            num_words = len(words)
            avg_word_length = num_chars / max(num_words, 1)

            # 2. Conteo de signos de puntuación
            num_exclamation = text.count('!')
            num_question = text.count('?')
            num_ellipsis = text.count('...') + text.count('..')
            num_caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 1)

            # 3. Ratio de mayúsculas
            num_upper_chars = sum(1 for c in text if c.isupper())
            ratio_upper = num_upper_chars / max(num_chars, 1)

            # 4. Presencia de palabras emocionales clave (one-hot por emoción)
            emotion_presence = {}
            for emotion, keywords in self.emotion_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                emotion_presence[emotion] = min(count, 3)  # Cap at 3 para evitar outliers

            # 5. Intensificadores
            num_intensifiers = sum(1 for word in words if word in self.intensifiers)

            # 6. Features adicionales
            has_negation = any(word in text_lower for word in ['not', 'no', 'never', "n't", 'dont', "don't"])
            has_first_person = any(word in words for word in ['i', 'me', 'my', 'mine', 'myself'])
            has_second_person = any(word in words for word in ['you', 'your', 'yours', 'yourself'])

            # Construir vector de features
            feature_vector = [
                num_chars,
                num_words,
                avg_word_length,
                num_exclamation,
                num_question,
                num_ellipsis,
                num_caps_words,
                ratio_upper,
                num_intensifiers,
                int(has_negation),
                int(has_first_person),
                int(has_second_person),
                emotion_presence['joy'],
                emotion_presence['sadness'],
                emotion_presence['anger'],
                emotion_presence['fear'],
                emotion_presence['love'],
                emotion_presence['surprise']
            ]

            features.append(feature_vector)

        return np.array(features)

    def get_feature_names(self):
        """Retorna los nombres de las features para interpretabilidad"""
        return [
            'num_chars',
            'num_words',
            'avg_word_length',
            'num_exclamation',
            'num_question',
            'num_ellipsis',
            'num_caps_words',
            'ratio_upper',
            'num_intensifiers',
            'has_negation',
            'has_first_person',
            'has_second_person',
            'joy_keywords',
            'sadness_keywords',
            'anger_keywords',
            'fear_keywords',
            'love_keywords',
            'surprise_keywords'
        ]


def test_feature_extractor():
    """Función de prueba para el extractor de features"""
    extractor = TextFeatureExtractor()

    test_texts = [
        "I am so happy today!",
        "I feel sad and lonely...",
        "I HATE THIS SO MUCH!!!",
        "I'm scared of what might happen",
        "I love you with all my heart",
        "WOW! I can't believe this happened!"
    ]

    features = extractor.transform(test_texts)
    feature_names = extractor.get_feature_names()

    print("Feature Extractor Test")
    print("=" * 80)
    for i, text in enumerate(test_texts):
        print(f"\nTexto: \"{text}\"")
        print(f"Features: {features[i]}")

    print(f"\n\nFeature names ({len(feature_names)}):")
    for name in feature_names:
        print(f"  - {name}")


if __name__ == '__main__':
    test_feature_extractor()
