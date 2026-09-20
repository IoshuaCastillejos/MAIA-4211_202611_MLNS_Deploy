import os.path as osp

import Definitions

# IMPORTANTE: este import va ANTES de joblib.load. Al importarlo se registra
# text_preprocess en __main__, que es donde pickle la busca al cargar el modelo.
from src.DataPreprocessing import DataPreprocessing

import joblib


class ModelController:

    def __init__(self):
        print("ModelController.__init__ ->")
        # Ruta del artefacto entrenado
        self.models_dir = osp.join(Definitions.ROOT_DIR, "resources/models")
        self.model_path = osp.join(self.models_dir, "model.joblib")

        # Un solo artefacto: el Pipeline completo del entrenamiento
        # (CountVectorizer -> TfidfTransformer -> TruncatedSVD -> LogisticRegression).
        # No hay pca.joblib ni scaler.joblib: esos pasos ya van dentro.
        self.model = joblib.load(self.model_path)

        # Clase de preprocesamiento de la información
        self.d_processing = DataPreprocessing()

    def get_categories(self):
        """Numeros de ODS que el modelo puede predecir."""
        return [int(c) for c in self.model.named_steps['classifier'].classes_]

    def predict(self, texto):
        """Recibe el texto CRUDO. El Pipeline se encarga del preprocesamiento."""
        print("ModelController.predict ->")
        ods = int(self.model.predict([texto])[0])
        nombre = self.d_processing.get_cat_name(ods)
        return ods, nombre

    def predict_proba(self, texto):
        """Devuelve [(ods, nombre, probabilidad), ...] de mayor a menor."""
        print("ModelController.predict_proba ->")
        probas = self.model.predict_proba([texto])[0]
        clases = self.get_categories()
        res = [
            (ods, self.d_processing.get_cat_name(ods), float(p))
            for ods, p in zip(clases, probas)
        ]
        return sorted(res, key=lambda x: x[2], reverse=True)
