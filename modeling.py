
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


def model_info():
    white=pd.read_csv('winequality-white.csv')
    red=pd.read_csv('winequality-red.csv')
    red['color']= 'red'
    white['color']= 'white'
    wine= pd.concat([red, white], ignore_index=True)
    wine, fences=remove_outliers(wine)
    wine=scale_wine(wine)
    wine=pd.get_dummies(wine, columns=['color'])
    train, val, test= tts(wine, 'quality')
    #setting values to cluster by
    X=train[['chlorides', 'residual sugar']]
    V=val[['chlorides', 'residual sugar']]
    #making, fitting, and predicting clusters
    kmeans = KMeans(n_clusters=4, random_state=8675309)
    kmeans.fit(X)

    train['rs_chl_cluster']=kmeans.predict(X)
    val['rs_chl_cluster']=kmeans.predict(V)

    #separating into x and y
    x_train= train.drop(columns=['quality', 'chlorides', 'residual sugar'])
    y_train= train['quality']

    x_val= val.drop(columns=['quality', 'chlorides', 'residual sugar'])
    y_val= val['quality']

    metrics= []

                                                        #build the model
    rm= RandomForestClassifier(max_depth= 2, min_samples_leaf= 1, random_state=8675309)
                                                        #fit the model
    rm.fit(x_train, y_train)
                                                        #get accuracy from in and out of sample data
    in_sample= rm.score(x_train, y_train)
    out_of_sample= rm.score(x_val, y_val)
                                                        #assigning the output to a dictionary
    output={'max_depth': 2,
            'min_samples_leaf': 1,
            'train_accuracy': in_sample,
            'validate_accuracy': out_of_sample,
            'cluster': 'residual_sugar_and_chloride'
            }
                                                        #appending the output dictionary to the empty metrics list
    metrics.append(output)

    train=train.drop(columns=['rs_chl_cluster'])
    val=val.drop(columns=['rs_chl_cluster'])
    ############################################################################################
    seed = 8675309
    X=train[['fixed acidity', 'volatile acidity']]
    V=val[['fixed acidity', 'volatile acidity']]
    kmeans= KMeans(n_clusters = 3, random_state = seed)

    kmeans.fit(X)

    train['bart_cluster']=kmeans.predict(X)
    val['bart_cluster']=kmeans.predict(V)

    #separating into x and y
    x_train= train.drop(columns=['quality', 'fixed acidity', 'volatile acidity'])
    y_train= train['quality']

    x_val= val.drop(columns=['quality', 'fixed acidity', 'volatile acidity'])
    y_val= val['quality']


                                                        #build the model
    rm= RandomForestClassifier(max_depth= 2, min_samples_leaf= 1)
                                                        #fit the model
    rm.fit(x_train, y_train)
                                                        #get accuracy from in and out of sample data
    in_sample= rm.score(x_train, y_train)
    out_of_sample= rm.score(x_val, y_val)
                                                        #assigning the output to a dictionary
    output={
        'max_depth': 2,
        'min_samples_leaf': 1,
        'train_accuracy': in_sample,
        'validate_accuracy': out_of_sample,
        'cluster': 'fixed_acidity_and_volatile_acidity'
    }
                                                        #appending the output dictionary to the empty metrics list
    metrics.append(output)

    ###################################################################################

    #setting values to cluster by
    X=train[['chlorides', 'residual sugar']]
    V=val[['chlorides', 'residual sugar']]
    #making, fitting, and predicting clusters
    kmeans = KMeans(n_clusters=4, random_state=8675309)
    kmeans.fit(X)

    train['rs_chl_cluster']=kmeans.predict(X)
    val['rs_chl_cluster']=kmeans.predict(V)

    #separating into x and y
    x_train= train.drop(columns=['quality', 'fixed acidity', 'volatile acidity', 'chlorides', 'residual sugar'])
    y_train= train['quality']

    x_val= val.drop(columns=['quality', 'fixed acidity', 'volatile acidity', 'chlorides', 'residual sugar'])
    y_val= val['quality']


                                                        #build the model
    rm= RandomForestClassifier(max_depth= 2, min_samples_leaf= 1, random_state=seed)
                                                        #fit the model
    rm.fit(x_train, y_train)
                                                        #get accuracy from in and out of sample data
    in_sample= rm.score(x_train, y_train)
    out_of_sample= rm.score(x_val, y_val)
                                                        #assigning the output to a dictionary
    output={
        'max_depth': 2,
        'min_samples_leaf': 1,
        'train_accuracy': in_sample,
        'validate_accuracy': out_of_sample,
        'cluster': 'both'
    }
                                                        #appending the output dictionary to the empty metrics list
    metrics.append(output)

    #############################################################################################

    train=train.drop(columns=['rs_chl_cluster', 'bart_cluster'])
    val=val.drop(columns=['rs_chl_cluster', 'bart_cluster'])

    #separating into x and y
    x_train= train.drop(columns=['quality'])
    y_train= train['quality']

    x_val= val.drop(columns=['quality'])
    y_val= val['quality']

                                                        #build the model
    rm= RandomForestClassifier(max_depth= 3, min_samples_leaf= 1, random_state=8675309)
                                                        #fit the model
    rm.fit(x_train, y_train)
                                                        #get accuracy from in and out of sample data
    in_sample= rm.score(x_train, y_train)
    out_of_sample= rm.score(x_val, y_val)
                                                        #assigning the output to a dictionary
    output={
        'max_depth': 3,
        'min_samples_leaf': 1,
        'train_accuracy': in_sample,
        'validate_accuracy': out_of_sample,
        'cluster': 'none'
    }
                                                        #appending the output dictionary to the empty metrics list
    metrics.append(output)
    metrics=pd.DataFrame(data=metrics)
    metrics['difference']=metrics['train_accuracy']-metrics['validate_accuracy']
    return metrics



def model_viz(df):
    plt.figure(figsize=(10,10))
    X = ['Cluster 1','Cluster 2','Both','None']
    trainacc = df['train_accuracy']
    valacc = df['validate_accuracy']
    diff= df['difference']
  
    X_axis = np.arange(len(X))
  
    plt.bar(X_axis - 0.2, trainacc, 0.4, label = 'Train Accuracy', color=['blue'], ec='black')
    plt.bar(X_axis + 0.2, valacc, 0.4, label = 'Validate Accuracy', color=['green'], ec='black')
  
    plt.xticks(X_axis, X)
    plt.xlabel("Model Includes")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Models")
    plt.ylim(0,.7)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.show()