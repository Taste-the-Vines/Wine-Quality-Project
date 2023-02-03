
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler

from prepare import remove_outliers, tts, scale_wine
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def mod_prep():
    white=pd.read_csv('winequality-white.csv')
    red=pd.read_csv('winequality-red.csv')
    red['color']= 'red'
    white['color']= 'white'
    wine= pd.concat([red, white], ignore_index=True)
    wine, fences=remove_outliers(wine)
    wine=scale_wine(wine)
    wine=pd.get_dummies(wine, columns=['color'])
    train, val, test= tts(wine, 'quality')
    
    x_train= train.drop(columns=['quality'])
    y_train= train['quality']

    x_val= val.drop(columns=['quality'])
    y_val= val['quality']

    x_test= test.drop(columns=['quality'])
    y_test= test['quality']
    return x_train, y_train, x_val, y_val, x_test, y_test


def cluster_1(train):
    X=train[['chlorides', 'residual sugar']]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    train['cluster']=kmeans.predict(X)
    
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    sns.relplot(data=train, x='chlorides', y='residual sugar', hue='cluster')
    centroids.plot.scatter(y='residual sugar', x='chlorides', c='black', marker='x', s=1000, 
                           ax=plt.gca(), label='centroid')
    plt.xlabel('Chlorides')
    plt.ylabel('Residual Sugar')
    plt.title('Residual Sugar/Chlorides Clusters')

    plt.show()


def cluster_2(train):
    seed = 8675309
    kmeans_scale = KMeans(n_clusters = 3, random_state = seed)
    X=train[['fixed acidity', 'volatile acidity']]
    kmeans_scale.fit(train[['fixed acidity', 'volatile acidity']])
    
    centroids = pd.DataFrame(kmeans_scale.cluster_centers_, columns=['fixed acidity', 'volatile acidity'])
    
    train['cluster']=kmeans_scale.predict(X)

    sns.relplot(data = train, x = 'volatile acidity', y = 'fixed acidity', hue = 'cluster')
    centroids.plot.scatter(y='fixed acidity', x='volatile acidity', 
                           c='black', marker='x', s=1000, ax=plt.gca(), label='centroid')
    plt.xlabel('Volatile Acidity')
    plt.ylabel('Fixed Acidity')
    plt.title('Volatile Acidity/Fixed Acidity Clusters')
    plt.show()


def baseline(train, y_train):
    train['base']= 6
    acc=accuracy_score(y_train,train['base'])
    print(f'Baseline accuracy is {round(acc,2)}')


def model(x_train, y_train, x_val, y_val, x_test, y_test):
    metrics= []
    rm= RandomForestClassifier(max_depth= 3, min_samples_leaf= 1, random_state=8675309)
    rm.fit(x_train, y_train)
    in_sample= rm.score(x_train, y_train)
    out_of_sample= rm.score(x_val, y_val)
    test=rm.score(x_test, y_test)
    output={'max_depth': 3,
            'min_samples_leaf': 1,
            'train_accuracy': in_sample,
            'validate_accuracy': out_of_sample,
            'test_accuracy': test
           }
    metrics.append(output)
    metrics=pd.DataFrame(data=metrics)
    return metrics


