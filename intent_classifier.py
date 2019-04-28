import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


class IntentClassifier:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict_intent(self, query):
        return 'CHITCHAT' if self.model.predict([query])[0] == 0 else 'IR'