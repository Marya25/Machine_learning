# Readme

## Projet : Classification de patients selon le type de cancer
### Introduction
Ce projet � pour but la comparaison de 5 algorithmes de classification supervis�e dans le but de cr�er le mod�le le plus adapt� pour la clasification de patients selon leur type de cancer � partir d'expression de g�nes (mesur�e par RNA-seq).

Les algorithmes compar�s sont:

- Naive Bayes
- K-nearest neighbor
- Random Forest
- Logistic Regression
- Decision Tree

Les classes de cancers etudi�es sont :

* Breast invasive carcinoma (BRCA)
* Kidney renal clear cell carcinoma (KIRC)
* Lung adenocarcinoma (LUAD)
* Prostate adenocarcinoma (PRAD)
* Colon adenocarcinoma (COAD)

### Pr�requis

Ce script � �t� d�velopp� avec Python 3.7

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
    + output_file.txt contient les r�sultats fourni par le script. Obtenu en redirigeant la sortie de la console par la commande suivante :
    >python main.py > output_file.txt
* main.py : script Python � executer
* Rapport.pdf : rapport su l'�tude r�aliser 

Les donn�es ont �t� obtenus � partir du UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

Auteure : Far Marya


