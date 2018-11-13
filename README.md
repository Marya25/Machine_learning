# Readme

## Projet : Classification de patients selon le type de cancer
### Introduction
Ce projet à pour but la comparaison de 5 algorithmes de classification supervisée dans le but de créer le modèle le plus adapté pour la clasification de patients selon leur type de cancer à partir d'expression de gènes (mesurée par RNA-seq).

Les algorithmes comparés sont:

- Naive Bayes
- K-nearest neighbor
- Random Forest
- Logistic Regression
- Decision Tree

Les classes de cancers etudiées sont :

* Breast invasive carcinoma (BRCA)
* Kidney renal clear cell carcinoma (KIRC)
* Lung adenocarcinoma (LUAD)
* Prostate adenocarcinoma (PRAD)
* Colon adenocarcinoma (COAD)

### Prérequis

Ce script à été développé avec Python 3.7

#### Librairies

* sklearn
* matplotlib
* pandas

### Fichiers
* Data  
    + data.csv : la table d'expression de gene par ecahntillon 
    + labels.csv : classe de cancer
* Output_files
    + Learning curves (format .png)obtenues pour chaque algorithme
    + output_file.txt contient les résultats fourni par le script. Obtenu en redirigeant la sortie de la console par la commande suivante :
    >python main.py > output_file.txt
* main.py : script Python à executer
* Rapport.pdf : rapport su l'étude réaliser 

Les données ont été obtenus à partir du UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

Auteure : Far Marya


