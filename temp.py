import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

#%%
df = pd.read_csv(r"D:\pythonProject\donnéesRevue.csv")

df = df.drop('translation', axis=1)
print("Taille du dataset (lignes, colonnes):", df.shape)
print("\n--- Aperçu des 5 premières lignes ---")
print(df.head())

print("\n--- Noms de colonnes ---")
print(df.columns)

print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())

if "label" in df.columns or "sentiment" in df.columns:
    col = "label" if "label" in df.columns else "sentiment"
    print("\n--- Distribution des classes ---")
    print(df[col].value_counts())

#%% apperçu des valeurs
print("\n--- Aperçu des valeurs uniques dans rating ---")
print(df["rating"].unique())

print("\n--- Distribution des notes ---")
print(df["rating"].value_counts())

#%%
df["review_length"] = df["review"].apply(lambda x: len(x.split()))

print(df["review_length"].describe())
print(df["review_length"].head(10))

#%% visualisation des longueurs des mots en graphes
plt.figure(figsize=(10, 6))
plt.hist(df["review_length"], bins=50, edgecolor='black')
plt.title("Distribution des longueurs des reviews")
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre de reviews")
plt.show()

#%%
min_length = 2
max_length = 200

df_clean = df[(df["review_length"] >= min_length) & (df["review_length"] <= max_length)]

print("Nouvelle taille du dataset :", df_clean.shape)

#%%
class_counts = df_clean["rating"].value_counts()
print(class_counts)

#%% nettoyage
stopwords_fr = set(stopwords.words('french'))
negation = {"ne", "pas", "jamais", "rien", "aucun"}
stopwords_fr = stopwords_fr - negation

def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"n['’]([a-zàâçéèêëîïôûùüÿñæœ]+)", r"ne \1", texte)
    texte = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", texte)
    mots = texte.split()
    mots = [mot for mot in mots if mot not in stopwords_fr]
    texte = " ".join(mots)
    return texte

df_clean["cleanedReview"] = df_clean["review"].apply(nettoyer_texte)
print(df_clean[["review", "cleanedReview"]].head(10))

#%% WordCloud français
texte_complet = " ".join(df_clean["cleanedReview"].astype(str))
wordcloud = WordCloud(width=800, height=400,
                      background_color="white",
                      colormap="viridis",
                      max_words=50).generate(texte_complet)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud des mots les plus fréquents (français)")
plt.show()

#%%
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_clean["cleanedReview"].astype(str))
y = df_clean["rating"]
print("Taille des features :", X)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
model = LogisticRegression(max_iter=1000)

#%%
model.fit(X_train, y_train)

#%%
y_pred = model.predict(X_test)

#%%
print("=== Évaluation du modèle ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nRapport de classification :\n", classification_report(y_test, y_pred))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

#%% Optimisation
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
log_reg = LogisticRegression(max_iter=1000)
grid = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print("Meilleurs hyperparamètres :", grid.best_params_)
print("Meilleure précision :", grid.best_score_)

best_model = grid.best_estimator_

#%% Sauvegarde
joblib.dump(best_model, "logistic_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Modèle et TF-IDF sauvegardés avec succès !")

#%%
model = joblib.load("logistic_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def predire_sentiment(commentaire):
    commentaire_nettoye = nettoyer_texte(commentaire)
    vecteur = tfidf.transform([commentaire_nettoye])
    prediction = model.predict(vecteur)
    return "Positif" if prediction[0] == 1 else "Négatif"

print(predire_sentiment(""))
