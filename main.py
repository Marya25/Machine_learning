# -*- coding: utf-8 -*-

#importing the libraries

# scipy
#import scipy as sc
# numpy
#import numpy as np
# matplotlib
import matplotlib as mpl
# pandas
import pandas as pd

# statsmodels
#import statsmodels as sm
# scikit-learn
import sklearn as sk
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

 #importing our dataset
dataset = pd.read_csv('Data\\data.csv' )

#importing cancer classes for each sample 
labels = pd.read_csv('Data\\labels.csv' )
#Adding the Class column from labels dataframe to dataset dataframe (to infome of the cancer class of the sample)
df = pd.concat([dataset, labels.Class],axis = 1)


def data_setting():
    print("*"*50, "\nData presentation","\n*"+"*"*49)
   
    dataset.shape
    nb_features = dataset.shape[1]-1
    nb_samples = dataset.shape[0]
    nb_cancer = labels.Class.value_counts() # gives the number of each class of cancer (5 differents types)
    df_head = df.head()
       
    #Checking for null and NA values
    nbNull = df.isnull().sum().sum()
    nbNa = df.isna().sum().sum()
    
      
    
    print("Number of samples: ",nb_samples,  "\nNumber of genes: ",nb_features, 
          "\n\nProportion of each cancer class:\n",nb_cancer, "\n\nFirst lines of the dataset:\n",
          df_head, "\n\nPresence of Na and Null values\nNumber of Na: ",nbNa, "\nNumber of Null: ",nbNull) 
    return df
   
    

#Stocking the features in X and the classes in Y

y = df.iloc[:,-1]
X = df.iloc[:,1:-1]



###################################################################

#Finding the best features number for 5 algorithm
k_list = [10, 50, 100, 200, 500, 1000, 10000, 20531] #♣Number of feautures tested

#Finding the best features number to keep for Naives Bayes model

def bestK_NB():
    print("*"*50, "\nFeatures selection","\n*"+"*"*49)
    print("\nNaive Bayes feature selection results")
    
    for i in k_list:
        
        #Features selection 
        selector = SelectKBest(chi2, k=i)
        selector.fit(X,y)    
        selectedFeatures = X.columns[selector.get_support(indices=True)].tolist() # Selected genes
        
        # New dataset with only selected genes
        dfSelectedFeature = pd.concat([df[selectedFeatures], labels.Class],axis = 1) 
            
        #Kept features in Xf
        Xf = dfSelectedFeature.iloc[:,:-1]
            
        #Creating Train and Test sets
        X_train, X_test, Y_train, Y_test = train_test_split(Xf, y, test_size = 0.30, random_state = 0)
        
        #Feature Scaling    
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        #Fitting Naive_Bayes
        classifier = GaussianNB()
        classifier.fit(X_train, Y_train)
        
        #Predicting the Test set results
        Y_pred = classifier.predict(X_test)
        
        #Evaluate the model
        print( "Number of features used: ",i, 
              "\tModel accuracy: ", metrics.accuracy_score(Y_test,Y_pred)) #Calculate accuracy
        

    
############################################################################
    
#Finding the best features number to keep for Decision tree 

def bestK_DecisionTree():
    print("\nDecision Tree feature selection results")
    for i in k_list:
        
        #Features selection 
        selector = SelectKBest(chi2, k=i)
        selector.fit(X,y)    
        selectedFeatures = X.columns[selector.get_support(indices=True)].tolist() # Selected genes
        
        # New dataset with only selected genes
        dfSelectedFeature = pd.concat([df[selectedFeatures], labels.Class],axis = 1) 
            
        #Kept features in Xf
        Xf = dfSelectedFeature.iloc[:,:-1]
            
        #Creating Train and Test sets
        X_train, X_test, Y_train, Y_test = train_test_split(Xf, y, test_size = 0.30, random_state = 0)
        
        #Feature Scaling    
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        #Fitting Decision Tree Algorithm
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, Y_train)
        
        #Predicting the Test set results
        Y_pred = classifier.predict(X_test)
        
        #Evaluate the model
        print( "Number of features used: ",i, 
              "\tModel accuracy: ", metrics.accuracy_score(Y_test,Y_pred)) #Calculate accuracy
        

    
#####################################################################


#Finding the best features number to keep for K-NN Algorithm

def bestK_KNN():
    print("\nK nearest neighbor feature selection results")
    for i in k_list:
        
        #Features selection 
        selector = SelectKBest(chi2, k=i)
        selector.fit(X,y)    
        selectedFeatures = X.columns[selector.get_support(indices=True)].tolist() # Selected genes
        
        # New dataset with only selected genes
        dfSelectedFeature = pd.concat([df[selectedFeatures], labels.Class],axis = 1) 
            
        #Kept features in Xf
        Xf = dfSelectedFeature.iloc[:,:-1]
            
        #Creating Train and Test sets
        X_train, X_test, Y_train, Y_test = train_test_split(Xf, y, test_size = 0.30, random_state = 0)
        
        #Feature Scaling    
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        #Fitting K-NN Algorithm
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, Y_train)
        
        #Predicting the Test set results
        Y_pred = classifier.predict(X_test)
        
        #Evaluate the model
        print( "Number of features used: ",i, 
              "\tModel accuracy: ", metrics.accuracy_score(Y_test,Y_pred)) #Calculate accuracy



#####################################################################


#Finding the best features number to keep for Logistic Regression


def bestK_LogReg():
    print("\nLogistic regression feature selection results")
    for i in k_list:
        
        #Features selection 
        selector = SelectKBest(chi2, k=i)
        selector.fit(X,y)    
        selectedFeatures = X.columns[selector.get_support(indices=True)].tolist() # Selected genes
        
        # New dataset with only selected genes
        dfSelectedFeature = pd.concat([df[selectedFeatures], labels.Class],axis = 1) 
            
        #Kept features in Xf
        Xf = dfSelectedFeature.iloc[:,:-1]
            
        #Creating Train and Test sets
        X_train, X_test, Y_train, Y_test = train_test_split(Xf, y, test_size = 0.30, random_state = 0)
        
        #Feature Scaling    
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        #Fitting the Logistic Regression Algorithm to the Training Set
        classifier = LogisticRegression()
        classifier.fit(X_train, Y_train)
        
        #Predicting the Test set results
        Y_pred = classifier.predict(X_test)
        
        #Evaluate the model
        print( "Number of features used: ",i, 
              "\tModel accuracy: ", metrics.accuracy_score(Y_test,Y_pred)) #Calculate accuracy
    
 
#####################################################################

#Finding the best features number to keep for Random Forest model

def bestK_RandomForest():
    print("\nRandom forest feature selection results")
    for i in k_list:
        
        #Features selection 
        selector = SelectKBest(chi2, k=i)
        selector.fit(X,y)    
        selectedFeatures = X.columns[selector.get_support(indices=True)].tolist() # Selected genes
        
        # New dataset with only selected genes
        dfSelectedFeature = pd.concat([df[selectedFeatures], labels.Class],axis = 1) 
            
        #Kept features in Xf
        Xf = dfSelectedFeature.iloc[:,:-1]
            
        #Creating Train and Test sets
        X_train, X_test, Y_train, Y_test = train_test_split(Xf, y, test_size = 0.30, random_state = 0)
        
        #Feature Scaling    
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        #Fitting Random Forest Classification Algorithm
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, Y_train)
        
        #Predicting the Test set results
        Y_pred = classifier.predict(X_test)
        
        #Evaluate the model
        print( "Number of features used: ",i, 
              "\tModel accuracy: ", metrics.accuracy_score(Y_test,Y_pred)) #Calculate accuracy


###################################################################

#Feature selection with the estmited best K
selector = SelectKBest(chi2, k=75)
selector.fit(X,y)
selectedFeatures = X.columns[selector.get_support(indices=True)].tolist() # Selected genes

# New dataset with only selected genes
dfSelectedFeature = pd.concat([df[selectedFeatures], labels.Class],axis = 1) 

#Kept features stocked in Xf
Xf = dfSelectedFeature.iloc[:,:-1]

#Creating Train and Test sets
X_train, X_test, Y_train, Y_test = train_test_split(Xf, y, test_size = 0.30, random_state = 0)
# print ("Train :", X_train.shape, Y_train.shape)
# print ("Test :", X_test.shape, Y_test.shape)

#Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

###################################################################


#
##Function to plot learning curves for the 5 chosen algorithm 
#def plot_learning_curve(algorithm, title, X, y, ylim=None, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
#   print("\n")
#   print("*"*50, "\nLearning curves","\n*"+"*"*49)
#   
#    plt.figure()
#    plt.title(title)
#    if ylim is not None:
#        plt.ylim(*ylim)
#    plt.xlabel("Number of training examples")
#    plt.ylabel("Model accuracy ")
#    
#    train_sizes, train_scores, test_scores = learning_curve(
#        algorithm, Xf, y, cv=cv,  train_sizes=train_sizes)
#    train_scores_mean = np.mean(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    
#    plt.grid()
#    plt.plot(train_sizes, train_scores_mean, 'o-', color='tab:pink',
#             label="Training")
#    plt.plot(train_sizes, test_scores_mean, 'o-', color='tab:blue',
#             label="Cross-validation")
#
#    plt.legend(loc="best")
#    return plt
#
#
##Cross validation with 100 iterations to get smoother mean test and train
##score curves, each time with 30% data randomly selected as a validation set
#cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)
#
##Naive Bayes Learning curves
#title = "Learning Curves (Naive Bayes)"
#algorithm = GaussianNB()
#plot_learning_curve(algorithm, title, Xf, y, ylim=(0.7, 1.01), cv=cv)
#
#
##K-NN learning curves
#title = "Learning Curves (K-NN)"
#algorithm = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#plot_learning_curve(algorithm, title, Xf, y, ylim=(0.95, 1.01), cv=cv)
#
#
##Random Forest learning curves
#title = "Learning Curves (Random Forest)"
#algorithm = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#plot_learning_curve(algorithm, title, Xf, y, ylim=(0.95, 1.02), cv=cv)
#
#
##Decision Tree learning curves
#title = "Learning Curves (Decision tree)"
#algorithm = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#plot_learning_curve(algorithm, title, Xf, y, ylim=(0.7, 1.01), cv=cv)
#
##Logistic Regression learning curves
#title = "Learning Curves (Logistic regression)"
#algorithm = LogisticRegression()
#plot_learning_curve(algorithm, title, Xf, y, ylim=(0.95, 1.01), cv=cv)


###################################################################

#Based on previous results the chosen model for the dataset is the Logistic regression


def predict_model():
    print("*"*50, "\nPrediction model","\n*"+"*"*49)    
    
    #Train and Test set desciption
    Train_size = Y_train.shape[0]
    Test_size = Y_test.shape[0]    
    
    #Fitting the Logistic Regression Algorithm to the Training Set
    classifier1 = LogisticRegression()
    classifier1.fit(X_train, Y_train)
    
    #Predicting the Test set results
    Y_pred1 = classifier1.predict(X_test)
    
    # Evaluate the model
    crossTab = pd.crosstab(Y_test,Y_pred1)
    
    #Feature selection
    print("\n\nFeature selection\nFor the feature selection K= ", Xf.shape[1])
    print ("\nSelected genes list\n",selectedFeatures )
    print("\n\nTrain and Test set description\nTrain set size: ",Train_size,  
          "\nTest set size: ",Test_size)
    print("\n\nModel Evaluation\nCross table between predicted and true cancer class:\n",
          crossTab,"\nModel accuracy :", metrics.accuracy_score(Y_test,Y_pred1))




data_setting()
bestK_NB()
bestK_DecisionTree()
bestK_KNN()
bestK_LogReg()
bestK_RandomForest()
predict_model()






