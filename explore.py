import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from prepare import remove_outliers, tts
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def get_wine():
    white=pd.read_csv('winequality-white.csv')
    red=pd.read_csv('winequality-red.csv')
    red['color']= 'red'
    white['color']= 'white'
    wine= pd.concat([red, white], ignore_index=True)
    return wine

def target_viz(df):
    sns.histplot(x='quality', data=df)
    plt.xlabel('Quality Rating')
    plt.title('Visualizing the Target Variable')
    plt.grid(True, alpha=0.3, linestyle='--')

def q1_plots(train):
    q3= train[train['quality']<6]
    q9= train[train['quality']>6]
    
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    sns.histplot(x='alcohol', data=q3)
    plt.title('Low Quality Wine (3-5)')
    plt.xlabel('Alcohol')
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.subplot(222)
    sns.histplot(x='alcohol', data=q9)
    plt.title('High Quality Wine (7-9)')
    plt.xlabel('Alcohol')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.figure(figsize=(25,10))
    plt.subplot(223)
    plt.title('High and Low Quality Wines')
    sns.histplot(x='alcohol', data=q3, alpha=.5, color='green', label= 'Low Quality')
    sns.histplot(x='alcohol', data=q9, alpha=.25, label='High Quality')
    plt.xlabel('Alcohol')
    plt.axvline(x=(q3['alcohol'].mean()), color='red', label='Low Quality Mean')
    plt.axvline(x=(q9['alcohol'].mean()), color='yellow', label='High Quality Mean')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.subplots_adjust(left=0.1,
                            bottom=-0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
    plt.show()

def q1_stat(df):
    alpha=0.05
    q3= df[df['quality']<6]
    q9= df[df['quality']>6]
    q3m=q3['alcohol']
    q9m=q9['alcohol']
    t, p=stats.ttest_ind(q3m,q9m, equal_var=False, alternative='less')
    if p<alpha:
        print(f'The p-value of {p} is less than the alpha ({alpha}) so we can reject the null hypothesis!')
    else:
        print('The p-value is greater than the alpha, so we can not reject the null hypothesis.')

def q2_plots(df):
    q9= df[df['quality']>6]
    
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    sns.histplot(x='chlorides', data=q9)
    plt.title('High Quality Wine (7-9)')
    plt.xlabel('Chlorides')
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.subplot(222)
    sns.histplot(x='chlorides', data=df)
    plt.title('All Wine')
    plt.xlabel('Chlorides')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.figure(figsize=(25,10))
    plt.subplot(223)
    plt.title('High/All Quality Wines')
    sns.histplot(x='chlorides', data=df, alpha=.5, color='green', label= 'All Wines')
    sns.histplot(x='chlorides', data=q9, alpha=.75, label='High Quality Wines')
    plt.xlabel('Chlorides')
    plt.axvline(x=(df['chlorides'].mean()), color='red', label='All Wine Mean')
    plt.axvline(x=(q9['chlorides'].mean()), color='yellow', label='High Quality Mean')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.subplots_adjust(left=0.1,
                            bottom=-0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
    plt.show()

def q2_stat(df):
    alpha=0.05
    q9= df[df['quality']>6]
    chlmean=df['chlorides'].mean()
    t, p = stats.ttest_1samp(q9['chlorides'],chlmean)
    if p<alpha:
        print(f'The p-value of {p} is less than the alpha ({alpha}) so we can reject the null hypothesis!')
    else:
        print('The p-value is greater than the alpha, so we can not reject the null hypothesis.')


def q3_plots(df):
    q34 = df[df['quality'] < 6]
    q89 = df[df['quality'] > 6]
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    sns.histplot(x='citric acid', data=q34)
    plt.title('Low Quality Wine (3-5)')
    plt.xlabel('Citric Acid')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.subplot(222)
    sns.histplot(x='citric acid', data=q89)
    plt.title('High Quality Wine (7-9)')
    plt.xlabel('Citric Acid')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.figure(figsize = (25,10))
    plt.subplot(223)
    plt.title('High and Low Quality Wines')
    sns.histplot(x='citric acid', data=q34, alpha=.25, color='green', label= 'Low Quality')
    sns.histplot(x='citric acid', data=q89, alpha=.50, label='High Quality')
    plt.xlabel('Citric Acid')
    plt.axvline(x=(q34['citric acid'].mean()), color='red', label='Low Quality Mean')
    plt.axvline(x=(q89['citric acid'].mean()), color='yellow', label='High Quality Mean')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.subplots_adjust(left=0.1,
                            bottom=-0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
    plt.show()


def q3_stat(df):
    alpha=0.05
    q34 = df[df['quality'] < 6]
    q89 = df[df['quality'] > 6]
    t, p = stats.ttest_ind(q34['citric acid'], q89['citric acid'], equal_var=False, alternative='less')
    if p<alpha:
        print(f'The p-value of {p} is less than the alpha ({alpha}) so we can reject the null hypothesis!')
    else:
        print('The p-value is greater than the alpha, so we can not reject the null hypothesis.')


def q4_plots(df):
    q34 = df[df['quality'] < 6]
    q89 = df[df['quality'] > 6]
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    sns.histplot(x='pH', data=q34)
    plt.title('Low Quality Wine (3-5)')
    plt.xlabel('pH')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.subplot(222)
    sns.histplot(x='pH', data=q89)
    plt.title('High Quality Wine (7-9)')
    plt.xlabel('pH')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.figure(figsize = (25,10))
    plt.subplot(223)
    plt.title('High and Low Quality Wines')
    sns.histplot(x='pH', data=q34, alpha=.25, color='green', label= 'Low Quality Wine')
    sns.histplot(x='pH', data=q89, alpha=.50, label='High Quality Wine')
    plt.xlabel('pH')
    plt.axvline(x=(q34['pH'].mean()), color='red', label='Low Quality Mean')
    plt.axvline(x=(q89['pH'].mean()), color='yellow', label='High Quality Mean')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.subplots_adjust(left=0.1,
                            bottom=-0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
    plt.show()


def q4_stat(df):
    alpha=0.05
    q34 = df[df['quality'] < 6]
    q89 = df[df['quality'] > 6]
    t, p = stats.ttest_ind(q34['pH'], q89['pH'], equal_var=False, alternative='less')
    if p<alpha:
        print(f'The p-value of {p} is less than the alpha ({alpha}) so we can reject the null hypothesis!')
    else:
        print('The p-value is greater than the alpha, so we can not reject the null hypothesis.')
