# Analyse de Sentiments - Projet Académique

Un projet d'analyse de sentiments sur des critiques de films en français, développé dans le cadre de mes études universitaires pour approfondir mes connaissances en Python et en analyse de données.

## Description

Ce projet implémente une analyse de sentiments en utilisant des techniques de traitement du langage naturel (NLP) et d'apprentissage automatique pour classifier les émotions exprimées dans les critiques de films en français. 

## Données

Le jeu de données `donnéesRevue.csv` n'est pas inclus dans ce dépôt en raison de sa taille. Vous pouvez en télécharger depuis [IMDb Reviews](https://www.imdb.com/).

## Structure du Projet

```
├── App.py               # API Flask pour servir le modèle
├── temp.py             # Script principal d'entraînement
├── models/            # Modèles entraînés
└── utils/            # Utilitaires
```

## Technologies Utilisées

- Python 3.x
- Flask pour l'API REST
- Pandas pour la manipulation des données
- Scikit-learn pour le machine learning
- NLTK pour le traitement du langage naturel

## Installation

1. Cloner le repository
```bash
git clone https://github.com/Tsiarofy/Sentiment-Analysis
```

2. Télécharger le fichier de données et le placer à la racine du projet

3. Installer les dépendances Python
```bash
pip install -r requirements.txt
```

## Utilisation

1. Pour entraîner le modèle :
```bash
python temp.py
```

2. Pour lancer l'API :
```bash
python App.py
```

L'API sera accessible sur `http://localhost:5000/predict`

### Créer votre interface

L'API expose un endpoint POST `/predict` qui attend un JSON avec la structure suivante :
```json
{
    "commentaire": "Votre texte à analyser"
}
```

Exemple de réponse :
```json
{
    "sentiment": "Positif"
}
```

Vous pouvez créer votre propre interface (web, mobile, desktop) en utilisant cette API. Exemples d'utilisation :

- **Avec cURL :**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"commentaire":"Ce film était fantastique!"}'
```

- **Avec JavaScript :**
```javascript
fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        commentaire: 'Ce film était fantastique!'
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Note Académique

Ce projet a été développé dans un contexte éducatif pour explorer :
- Le traitement du langage naturel
- L'apprentissage automatique 
- L'analyse de données
- Le développement d'API REST

## Licence

MIT