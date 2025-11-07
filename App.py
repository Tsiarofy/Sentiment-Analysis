from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import sys


# Charger le modèle et le TF-IDF
model = joblib.load("D:/pythonProject/models/logistic_model.pkl")
tfidf = joblib.load("D:/pythonProject/models/tfidf_vectorizer.pkl")
sys.path.append("D:/pythonProject/utils") 

from nettoyage import nettoyer_texte  

app = Flask(__name__)
CORS(app)  # Autorise React à accéder à l’API

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    commentaire = data.get('commentaire', '')

   

    commentaire_nettoye = nettoyer_texte(commentaire)
    vecteur = tfidf.transform([commentaire_nettoye])
    prediction = model.predict(vecteur)[0]

    return jsonify({'sentiment': 'Positif' if prediction == 1 else 'Négatif'})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
