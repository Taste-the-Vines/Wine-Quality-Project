import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



#function for doing train test split on any continuous variable
def tts(df, stratify=None):
    '''
    removing your test data from the data
    '''
    train_validate, test=train_test_split(df, 
                                 train_size=.8, 
                                 random_state=8675309,
                                 stratify=None)
    '''
    splitting the remaining data into the train and validate groups
    '''            
    train, validate =train_test_split(train_validate, 
                                      test_size=.3, 
                                      random_state=8675309,
                                      stratify=None)
    return train, validate, test


#function to remove outliers
def remove_outliers(df, k=1.5):
    a=[]
    b=[]
    fences=[a, b]
    features= []
    col_list = []
    i=0
    for col in df:
        new_df=np.where(df[col].nunique()>8, True, False)
        if new_df==True:
            if df[col].dtype == 'float' or df[col].dtype == 'int':
                '''
                for each feature find the first and third quartile
                '''
                q1, q3 = df[col].quantile([.25, .75])
                '''
                calculate inter quartile range
                '''
                iqr = q3 - q1
                '''
                calculate the upper and lower fence
                '''
                upper_fence = q3 + (k * iqr)
                lower_fence = q1 - (k * iqr)
                '''
                appending the upper and lower fences to lists
                '''
                a.append(upper_fence)
                b.append(lower_fence)
                '''
                appending the feature names to a list
                '''
                features.append(col)
                '''
                assigning the fences and feature names to a dataframe
                '''
                var_fences= pd.DataFrame(fences, columns=features, index=['upper_fence', 'lower_fence'])
                col_list.append(col)
            else:
                print(col)
                print('column is not a float or int')
        else:
            print(f'{col} column ignored')
                                    
    for col in col_list:
        '''
        reassigns the dataframe to only include values withing the upper and lower fences/drop outliers
        '''
        df = df[(df[col]<= a[i]) & (df[col]>= b[i])]
        i+=1
    return df, var_fences


def missing_nulls(df):
    a=[]
    b=[]
    c=[]
    e= {'num_rows_missing': a, 'pct_rows_missing': b}
    for col in df:
        c.append(col)
        nulls= sum(df[col].isnull())
        a.append(nulls)
        b.append(nulls/len(df[col]))
        d= pd.DataFrame(index=c, data=e)
    return d

def handle_missing_values(df, prop_req_column, prop_req_row):
    drop_col= len(df)* prop_req_column
    df= df.dropna(axis=1, thresh= drop_col)
    drop_row= len(df.columns)* prop_req_row
    df= df.dropna(axis=0, thresh=drop_row)
    return df