#!/usr/bin/env python
# coding: utf-8

# # Modèle d'évaluation du niveau d'anglais

# In[46]:


# Librairies importées pour diverses tâches

import os  # Fournit des fonctions pour interagir avec le système d'exploitation, telles que la gestion des fichiers et des répertoires.
import openai  # Permet d'interagir avec les modèles d'OpenAI, comme GPT, pour générer du texte ou effectuer des tâches d'IA.
import re  # Utilisée pour effectuer des opérations de recherche et de manipulation de chaînes de caractères en utilisant des expressions régulières.
from collections import Counter  # Permet de compter les occurrences d'éléments dans un itérable, utile pour les statistiques de fréquence.
import csv  # Fournit des outils pour lire et écrire des fichiers CSV (Comma-Separated Values), utilisés pour stocker des données sous forme tabulaire.
import pandas as pd  # Bibliothèque pour la manipulation et l'analyse de données sous forme de DataFrame, qui permet de traiter et d'analyser des structures de données complexes.
import textwrap  # Utilisée pour formater des chaînes de caractères en enveloppant le texte sur plusieurs lignes, utile pour l'affichage de textes longs.
from tabulate import tabulate  # Permet de formater et afficher des tableaux de manière lisible dans la console, utile pour visualiser des données sous forme tabulaire.
import matplotlib.pyplot as plt  # Bibliothèque de visualisation pour générer des graphiques et des plots, utile pour afficher des données visuellement.
import seaborn as sns  # Basée sur matplotlib, cette bibliothèque facilite la création de visualisations statistiques attrayantes et informatives.
from sklearn.feature_extraction.text import CountVectorizer  # Transforme les textes en vecteurs de comptage de mots, utile pour la préparation des données textuelles avant l'apprentissage automatique.
from sklearn.model_selection import train_test_split  # Permet de diviser les données en ensembles d'entraînement et de test pour l'évaluation des modèles.
from sklearn.ensemble import RandomForestClassifier  # Implémente un classifieur utilisant un ensemble d'arbres de décision pour effectuer des prédictions sur les données.
from sklearn.feature_extraction.text import TfidfVectorizer  # Convertit les textes en vecteurs numériques en utilisant la méthode TF-IDF (Term Frequency-Inverse Document Frequency), qui reflète l'importance des mots dans un document par rapport à un corpus.
from sklearn.metrics import precision_score, recall_score, f1_score  # Calcule respectivement la précision, le rappel et le F1-score pour évaluer les performances d'un modèle de classification.


# In[2]:


# Clé API OpenAI
openai.api_key = "maclé"


# In[3]:


# Fonction pour transcrire un fichier MP3 à l'aide de l'API OpenAI (Whisper)
def transcribe_mp3(file_path):
    """
    Transcrit un fichier audio MP3 en texte à l'aide de Whisper via l'API OpenAI.
    
    :param file_path: Chemin vers le fichier MP3.
    :return: Texte transcrit.
    """
    try:
        with open(file_path, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
        return response["text"]
    except Exception as e:
        raise RuntimeError(f"Erreur pendant la transcription audio: {e}")
#Nous avons eu à uutiliser l'API d'open AI "Wishper" pour faire de la transcription


# In[4]:


# Fonction pour pré-analyser une transcription pour identifier des erreurs courantes et des métriques clés
def preprocess_transcription(transcription):
    """
    Pré-analyse d'une transcription pour identifier des erreurs courantes et des métriques clés.
    
    :param transcription: Texte transcrit.
    :return: Analyse des erreurs et métriques clés.
    """
    analysis = {}
    
    # 1. Débit de parole
    word_count = len(transcription.split())
    analysis["word_count"] = word_count
    analysis["speech_rate"] = word_count / 60  # Supposons une durée de 60 secondes pour l'audio
    
    # 2. Analyse des répétitions
    words = transcription.lower().split()
    word_freq = Counter(words)
    repeated_words = [word for word, count in word_freq.items() if count > 3]
    analysis["repeated_words"] = repeated_words
    
    # 3. Erreurs grammaticales courantes
    grammar_issues = []
    common_errors = {
    r"\bI has\b": "Incorrect: 'I has' -> Correct: 'I have'",
    r"\bhe do\b": "Incorrect: 'He do' -> Correct: 'He does'",
    r"\bthere is many\b": "Incorrect: 'There is many' -> Correct: 'There are many'",
    r"\bmore better\b": "Incorrect: 'More better' -> Correct: 'Better'",
    r"\bI didn't went\b": "Incorrect: 'I didn't went' -> Correct: 'I didn't go'",
    r"\bshe don't\b": "Incorrect: 'She don't' -> Correct: 'She doesn't'",
    r"\bcan able to\b": "Incorrect: 'Can able to' -> Correct: 'Can'",
    r"\bI am agree\b": "Incorrect: 'I am agree' -> Correct: 'I agree'",
    r"\bme and my\b": "Incorrect: 'Me and my friend went' -> Correct: 'My friend and I went'",
    r"\bmore faster\b": "Incorrect: 'More faster' -> Correct: 'Faster'",
    r"\bamount of people\b": "Incorrect: 'Amount of people' -> Correct: 'Number of people'",
    r"\bless books\b": "Incorrect: 'Less books' -> Correct: 'Fewer books'",
    r"\badvices\b": "Incorrect: 'Advices' -> Correct: 'Advice' (uncountable noun)",
    r"\ba equipments\b": "Incorrect: 'A equipments' -> Correct: 'Equipment' (uncountable noun)",
    r"\ban information\b": "Incorrect: 'An information' -> Correct: 'Information' (uncountable noun)",
    r"\bthese kind of\b": "Incorrect: 'These kind of things' -> Correct: 'This kind of things'",
    r"\bsince\b.*?\b(years|months|days)\b": "Incorrect: 'Since 5 years' -> Correct: 'For 5 years'",
    r"\bI known him\b": "Incorrect: 'I known him' -> Correct: 'I have known him'",
    r"\bbring me to\b": "Incorrect: 'Bring me to the airport' -> Correct: 'Take me to the airport'",
    r"\bsomeone that\b": "Incorrect: 'Someone that' -> Correct: 'Someone who'",
    r"\bwho's\b": "Incorrect: 'Who's car is this?' -> Correct: 'Whose car is this?'"
}
    for pattern, correction in common_errors.items():
        if re.search(pattern, transcription):
            grammar_issues.append(correction)
    analysis["grammar_issues"] = grammar_issues
    
    return analysis

Cette fonction(def preprocess_transcription(transcription) est utile pour analyser automatiquement une transcription afin d'identifier des erreurs courantes, comme les fautes grammaticales et les répétitions, et pour calculer des métriques clés telles que le débit de parole. Son apport au prompt réside dans la capacité à fournir des informations détaillées sur la qualité linguistique et les performances du locuteur, ce qui peut enrichir l'évaluation et guider les recommandations.
# In[5]:


def generate_prompt(preprocessed_analysis, transcription):
    """
    Génère un prompt structuré pour GPT basé sur l'analyse automatique et un format détaillé,
    avec des catégories simplifiées pour le niveau d'anglais.
    """
    return f"""
    Tu es un évaluateur linguistique spécialisé dans l'analyse des compétences en anglais. 
    Analyse le texte fourni en identifiant, quantifiant et expliquant les erreurs selon ces catégories :

    1. *Fautes grammaticales* :
       - Conjugaison des verbes : Utilisation incorrecte des temps (e.g., "He go to school" au lieu de "He goes to school").
       - Accord sujet-verbe : Erreurs dans l'accord (e.g., "They is happy" au lieu de "They are happy").
       - Utilisation des articles : Omission ou mauvaise utilisation des articles définis ou indéfinis (e.g., "I saw a elephant" au lieu de "I saw an elephant").
       - Mauvais ordre des mots : Syntaxe incorrecte (e.g., "What you are doing?" au lieu de "What are you doing?").
       - Prépositions : Utilisation incorrecte (e.g., "I am good in English" au lieu de "I am good at English").
       Quantifie le nombre de fautes pour chaque type.

    2. *Fautes lexicales* :
       - Choix du mot : Mauvais choix de vocabulaire (e.g., "I am boring" au lieu de "I am bored").
       - Confusion entre mots proches : Homonymes ou synonymes mal utilisés (e.g., "accept" vs "except").
       - Faux amis : Mots qui ressemblent à une autre langue mais ont un sens différent (e.g., "actually" interprété comme "actuellement").
       Quantifie les fautes lexicales et propose des alternatives.

    3. *Fautes d'orthographe* :
       - Erreurs typographiques : (e.g., "definately" au lieu de "definitely").
       - Orthographe américaine vs britannique : (e.g., "color" vs "colour").
       - Omissions ou ajouts de lettres : (e.g., "recieve" au lieu de "receive").
       Quantifie les fautes détectées et indique leur proportion par rapport au texte.

    4. *Fautes de style et de registre* :
       - Usage inapproprié : Utilisation d’un ton informel dans un contexte formel.
       - Répétitions ou redondances : (e.g., "He is very very happy").
       Quantifie les occurrences et propose des corrections.

    5. *Fautes liées à la structure et à la cohérence* :
       - Organisation des idées : Analyse la clarté des paragraphes et des phrases.
       - Transitions insuffisantes : Analyse l’enchaînement des idées.
       Quantifie les transitions manquantes ou inadéquates.

    #### Sortie attendue :
    1. Rapport détaillé par critère : Fournis une liste des fautes détectées, des explications et des suggestions de correction.
    2. Tableau des scores résumé : Donne un score sur 10 pour chaque catégorie et un score final pondéré (en pourcentage).
       | Critère         | Score (/10) |
       |-----------------|-------------|
       | Grammaire       | -           |
       | Vocabulaire     | -           |
       | Orthographe     | -           |
       | Fluidité        | -           |
       | Cohérence       | -           |
       | Score final (%) | -           |
       | Niveau estimé   | -           |
    3. Niveau estimé global : Attribue l'un des niveaux suivants en fonction des performances générales :
       - **Débutant**
       - **Intermédiaire**
       - **Avancé**
    4. Recommandations : Fournis un résumé des axes d’amélioration.
    
    Texte à évaluer : {transcription}
    """


# In[6]:


# Fonction pour évaluer un fichier MP3
def evaluate_mp3(file_path):
    """
    Workflow complet : Transcrire, pré-analyser et évaluer un fichier MP3.
    
    :param file_path: Chemin vers le fichier MP3.
    :return: Résultat de l'évaluation.
    """
    try:
        transcription = transcribe_mp3(file_path)
        analysis = preprocess_transcription(transcription)
        prompt = generate_prompt(analysis, transcription)
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Vous êtes un évaluateur d'anglais selon le cadre CEFR."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            "file_name": os.path.basename(file_path),
            "transcription": transcription,
            "word_count": analysis['word_count'],
            "speech_rate": analysis['speech_rate'],
            "repeated_words": ', '.join(analysis['repeated_words']),
            "grammar_issues": '; '.join(analysis['grammar_issues']),
            "gpt_feedback": response['choices'][0]['message']['content']
        }
    except Exception as e:
        return {"error": str(e)}


# In[7]:


# Fonction pour évaluer tous les fichiers MP3 dans un répertoire donné et écrire les résultats dans un fichier CSV
def evaluate_mp3_files_in_directory(directory_path, output_csv_path):
    """
    Évalue tous les fichiers MP3 dans un répertoire donné et enregistre les résultats dans un fichier CSV.
    
    :param directory_path: Chemin vers le répertoire contenant les fichiers MP3.
    :param output_csv_path: Chemin vers le fichier CSV de sortie.
    """
    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(directory_path, filename)
            print(f"Traitement du fichier : {file_path}")
            result = evaluate_mp3(file_path)
            results.append(result)
    
    # Écriture des résultats dans un fichier CSV
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "transcription", "word_count", "speech_rate", "repeated_words", "grammar_issues", "gpt_feedback"])
        writer.writeheader()
        writer.writerows(results)

# Exemple d'utilisation avec un répertoire de fichiers MP3
directory_path = r"C:\Users\Chris\Music\fichiers audio clinique"  # Remplacez par le chemin de votre répertoire
output_csv_path = r"C:\Users\Chris\OneDrive\Documents\Mes données Pleiades\Sauvegarde des projets\evaluation_results.csv"  # Le chemin du fichier CSV de sortie
evaluate_mp3_files_in_directory(directory_path, output_csv_path)


# In[ ]:





# In[8]:


df=pd.read_csv(r"C:\Users\Chris\OneDrive\Documents\Mes données Pleiades\Sauvegarde des projets\evaluation_results.csv")


# In[9]:


df


# In[10]:


df.isna().sum()


# On observe ici qu'il y a 16 valeurs manquantes pour les colonnes "repeated_words" et "grammar_issues" car les données fournies sont très insuffisantes, celles-ci ne nous ont servies que de test afin d'avoir un aperçu de notre modèle

# In[11]:


df.shape


# In[12]:


df.iloc[9]


# In[13]:


# Fonction mise à jour pour inclure explicitement le niveau estimé
def extract_scores_recommendations_and_level(feedback):
    """
    Extrait les scores, le niveau estimé et les recommandations depuis le feedback.
    """
    # Extraction des scores avec une expression régulière robuste
    scores = re.findall(r"\| ([\w\s]+)\s*\|\s*(\d+|Non applicable|Non spécifié|C\d+)\s*\|", feedback)
    scores = [[score[0].strip(), score[1].strip()] for score in scores]
    
    # Vérification du niveau estimé dans le texte
    level_match = re.search(r"Niveau estimé\s+\|\s+([A-Z0-9]+)\s+\|", feedback)
    level = level_match.group(1).strip() if level_match else "Non spécifié"

    # Ajout explicite du niveau estimé si absent dans les scores
    if not any("Niveau estimé" in item[0] for item in scores):
        scores.append(["Niveau estimé", level])

    # Extraction des recommandations
    recommendations_match = re.search(r"Recommandations :\n(.+)", feedback, re.DOTALL)
    recommendations = recommendations_match.group(1).strip() if recommendations_match else "Aucune recommandation."

    return scores, recommendations


# In[14]:


# Fonction mise à jour pour afficher le feedback formaté
def display_feedback_from_df(df, row_idx):
    """
    Affiche un retour structuré avec une analyse détaillée, un tableau de scores et des recommandations.
    """
    # Récupérer le feedback pour la ligne spécifiée
    feedback = df.loc[row_idx, 'gpt_feedback']

    # Mise en forme du texte principal
    wrapper = textwrap.TextWrapper(width=80)
    formatted_feedback = "\n".join(wrapper.wrap(feedback.strip()))
    
    # Extraction automatique des scores, niveau estimé et recommandations
    scores, recommendations = extract_scores_recommendations_and_level(feedback)
    
    # Formatage du tableau des scores avec tabulate
    table = tabulate(scores, headers=["Critère", "Score (/10)"], tablefmt="fancy_grid")
    
    # Impression du contenu formaté
    print("\nRésumé complet :\n")
    print(formatted_feedback)
    print("\nTableau des scores :\n")
    print(table)
    print("\nRecommandations :\n")
    print(recommendations)


# In[15]:


# Appel de la fonction pour une ligne donnée (par exemple, ligne 9)
display_feedback_from_df(df,8)

Ici, il est à spécifier que plus l'audio est court plus notre modèle est en difficulté et n'arrive pas à sortir un resultat convenable
# In[ ]:





# In[16]:


# Exemple d'utilisation avec un répertoire de fichiers MP3
directory_path = r"C:\Users\Chris\Music\fichier audio non natifs" # Remplacez par le chemin de votre répertoire
output_csv_path = r"C:\Users\Chris\OneDrive\Documents\Mes données Pleiades\Sauvegarde des projets\df_non_natif.csv"  # Le chemin du fichier CSV de sortie
evaluate_mp3_files_in_directory(directory_path, output_csv_path)


# In[17]:


df_non_natif=pd.read_csv(r"C:\Users\Chris\OneDrive\Documents\Mes données Pleiades\Sauvegarde des projets\df_non_natif.csv")


# In[18]:


df_non_natif


# In[19]:


# Appel de la fonction pour une ligne donnée (par exemple, ligne 9)
display_feedback_from_df(df_non_natif, 3)


# In[20]:


df_non_natif['transcription'][7]


# In[21]:


# Fusionner les deux datasets
df_fusionne = pd.concat([df, df_non_natif], ignore_index=True)

# Vérifier la fusion
df_fusionne.info()


# In[22]:


# Extraire le niveau estimé depuis la colonne 'gpt_feedback'
y_true = df_fusionne['gpt_feedback'].str.extract(r'Niveau estimé\s*[:|-]?\s*(\S+)')

# Nettoyer et retirer les espaces
y_true = y_true[0].str.strip()

# Supprimer les lignes avec des valeurs manquantes
y_true = y_true.dropna()

# Vérifier les niveaux extraits
print(y_true.value_counts())


# In[58]:


# Remplacer "Débutante" par "Débutant" et "Intermediaire|" par "Intermediaire"
y_true = y_true.replace({
    "Débutante": "Débutant",
    "Intermediaire|": "Intermediaire",
    "-": None  # Supprimer les valeurs avec "-"
})

# Supprimer les lignes avec des valeurs manquantes
y_true = y_true.dropna()

# Vérifier les résultats après nettoyage
print(y_true.value_counts())


# In[24]:


# Supprimer les lignes contenant des valeurs manquantes dans X ou y
dataset_fusionne_cleaned = df_fusionne.dropna(subset=['transcription', 'gpt_feedback'])
df_fusionne.isna().sum()


# In[ ]:





# In[57]:


# Nettoyer y_true en remplaçant "Débutante" par "Débutant", et "Intermediaire|" par "Intermediaire"
y = df_fusionne['gpt_feedback'].str.extract(r'Niveau estimé\s*[:|-]?\s*(\S+)').fillna('')

# Appliquer les nettoyages supplémentaires
y = y.replace({
    "Débutante": "Débutant",
    "Intermediaire|": "Intermediaire",
    "-": None  # Supprimer les valeurs manquantes ou incohérentes
})

# Supprimer les lignes avec des valeurs manquantes dans y
y = y.dropna()

# S'assurer que X et y ont la même longueur
X = df_fusionne['transcription'].fillna('')
X = X.loc[y.index]  # Aligner X avec les indices de y

# Convertir les textes en vecteurs numériques avec TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_vect = vectorizer.fit_transform(X)

# Vérifier que X_vect et y ont le même nombre de lignes
print(X_vect.shape[0], len(y))  # Les longueurs devraient être identiques

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Entraîner un modèle RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test (classification)
y_pred = model_rf.predict(X_test)

# Comparer les prédictions avec les vrais niveaux
comparison = pd.DataFrame({'y_true': y_test.squeeze(), 'y_pred': y_pred})
print(comparison)

# Calcul des métriques de classification
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

# Affichage des métriques
print(f"Précision: {precision}")
print(f"Rappel: {recall}")
print(f"F1-Score: {f1}")

Précision : 0.75
La précision est de 75 %, ce qui signifie que, parmi toutes les prédictions faites par le modèle comme étant "Avancé", 75 % étaient correctes. Cela indique que le modèle est relativement bon pour prédire les cas "Avancé" mais pourrait commettre des erreurs sur d'autres catégories.

Rappel : 0.5
Le rappel est de 50 %, ce qui signifie que, parmi tous les vrais cas d'anglais "Avancé", seulement 50 % ont été correctement identifiés par le modèle. Cela suggère que le modèle manque une partie des "Avancé", ce qui peut indiquer une sous-performance dans la détection de cette catégorie.

F1-Score : 0.33
L'F1-score de 0,33, qui est la moyenne harmonique entre précision et rappel, est relativement faible. Cela indique qu'il y a un déséquilibre dans les performances du modèle, où il existe un compromis entre ces deux métriques. Ce score suggère qu'il y a encore des améliorations à apporter au modèle, en particulier pour mieux identifier la catégorie "Avancé" et réduire les erreurs de prédiction.
# In[50]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

print(f"Dimensions de X_train : {X_train.shape}")
print(f"Dimensions de y_train : {y_train.shape}")


# In[51]:


y_train = y_train.squeeze()
y_test = y_test.squeeze()


# In[53]:


model_rf.fit(X_train, y_train)


# In[62]:


df_fusionne['gpt_feedback'][15]


# In[ ]:




