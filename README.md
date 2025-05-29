# Évaluation Automatique du Niveau d'Anglais — Speech-to-Level

## Description

Ce projet a pour objectif d’évaluer automatiquement le niveau d’anglais parlé des participants à partir de leurs enregistrements audio. Le système analyse la transcription des discours ainsi que des caractéristiques vocales pour classer chaque locuteur dans une catégorie de niveau linguistique : **Débutant**, **Intermédiaire** ou **Avancé**.

L’évaluation s’appuie sur une combinaison d’analyses linguistiques (compte de mots, erreurs grammaticales, répétitions) et d’indicateurs prosodiques (rythme de parole, fluidité). Cette approche permet d’obtenir une estimation objective et rapide du niveau de maîtrise de l’anglais oral.

---

## Données utilisées

Pour entraîner et tester le modèle, j’ai utilisé deux datasets :

- **Dataset des natifs** : collecté à partir d’un site web anglophone (source externe).  
- **Dataset des non-natifs** : enregistrements réalisés par moi-même et quelques amis, dans le but de constituer un corpus représentatif.

Ces deux jeux de données m’ont permis de comparer les performances du modèle sur des voix natives et non-natives.

---

## Résultats obtenus

Les métriques d’évaluation actuelles sont les suivantes :

- **Précision** : 0.75  
- **Recall (rappel)** : 0.50  
- **F1-Score** : 0.30  

Ces résultats indiquent une performance modérée, probablement limitée par la qualité et la taille des données non-natives, qui restent relativement petites et peu diversifiées.

---

## Fonctionnalités principales

- **Extraction audio → texte** : traitement et transcription des fichiers audio soumis par les participants.  
- **Analyse linguistique** : détection de la qualité du discours, identification des erreurs, comptage de mots et répétitions.  
- **Analyse vocale** : calcul du rythme de parole, détection des pauses et fluidité.  
- **Classification automatique** : algorithme basé sur des règles et/ou machine learning pour attribuer un niveau de compétence (A1 à C2, regroupés en Débutant, Intermédiaire, Avancé).  
- **Rapport personnalisé** : pour chaque participant, génération d’un résumé détaillant points forts, axes d’amélioration, et niveau estimé.  

---

## Installation et utilisation

1. **Pré-requis** : Python 3.8+, bibliothèques `pandas`, `numpy`, `speech_recognition`, `scikit-learn`, etc.  
2. **Cloner ce dépôt** :  
   ```bash
   git clone https://github.com/tonpseudo/english-level-evaluation.git
   cd english-level-evaluation
