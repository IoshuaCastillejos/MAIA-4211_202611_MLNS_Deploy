import sys

import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

SW_ES = set(stopwords.words('spanish'))


def text_preprocess(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    tokens = [w for w in tokens if w not in SW_ES]
    return ' '.join(tokens)


# Pickle busca text_preprocess en __main__, no en este modulo.
sys.modules['__main__'].text_preprocess = text_preprocess

ODS_NOMBRES = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educacion de calidad",
    5: "Igualdad de genero",
    6: "Agua limpia y saneamiento",
    7: "Energia asequible y no contaminante",
    8: "Trabajo decente y crecimiento economico",
    9: "Industria, innovacion e infraestructura",
    10: "Reduccion de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Produccion y consumo responsables",
    13: "Accion por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones solidas",
}


class DataPreprocessing:

    def __init__(self):
        print("DataPreprocessing.__init__ ->")

    def get_cat_name(self, index):
        """Devuelve el nombre del ODS a partir de su numero."""
        return ODS_NOMBRES.get(int(index), f"ODS {index}")
