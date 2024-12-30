import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pickle
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier


def cleaned_data():
    df = pd.read_csv('data/data.csv')

    # Handling Missing Values
    missing_values_percent = df.isnull().mean()*100
    data_loss_cols = missing_values_percent[missing_values_percent>25].index

    df.drop(data_loss_cols, axis=1, inplace=True)

    missing_impute_cols = missing_values_percent[(missing_values_percent>10) & (missing_values_percent<25)]

    idx = missing_impute_cols.index
    num_miss_col = df[idx].select_dtypes(include=np.number).columns

    imputer = KNNImputer(n_neighbors=5)
    df[num_miss_col] = imputer.fit_transform(df[num_miss_col])
    
    df.dropna(inplace=True)
    df.drop(['id','Name','City'], axis=1, inplace=True)

    df[num_miss_col] = df[num_miss_col].astype(int)

    dietary_to_map = df['Dietary Habits'].value_counts()[df['Dietary Habits'].value_counts() < 100].index
    sleep_to_map = df['Sleep Duration'].value_counts()[df['Sleep Duration'].value_counts() < 100].index

    df['Dietary Habits'] = df['Dietary Habits'].replace(dietary_to_map, 'Other')
    df['Sleep Duration'] = df['Sleep Duration'].replace(sleep_to_map, 'Other')

    degrees_to_map = df['Degree'].value_counts()[df['Degree'].value_counts() < 1000].index
    df['Degree'] = df['Degree'].replace(degrees_to_map, 'Other')

    df['Age']=df['Age'].astype(int)
    df['Work/Study Hours']=df['Work/Study Hours'].astype(int)
    df['Financial Stress']=df['Financial Stress'].astype(int)
    

    print(df.head())
    return df

def preprocessing(df):
    catcols_high_cardinality = ['Sleep Duration', 'Dietary Habits', 'Degree']
    catcols_low_cardinality = ['Gender', 'Working Professional or Student', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

    ## Encoding categorical features
    oe = OrdinalEncoder()
    df[catcols_low_cardinality] = oe.fit_transform(df[catcols_low_cardinality])
    df[catcols_low_cardinality]=df[catcols_low_cardinality].astype(int)

    te = TargetEncoder(cols=df[catcols_high_cardinality])
    df[catcols_high_cardinality] = te.fit_transform(df[catcols_high_cardinality], df['Depression'])

    return df,te,oe


def trained_model(df):
    features = df.drop('Depression', axis=1)
    target = df['Depression']

    rf = RandomForestClassifier(n_estimators=200,max_depth=10)
    rf.fit(features, target)
    
    return rf


def main():
    clean_data = cleaned_data()
    with open('model/cleaned_data.pkl','wb') as file:
        pickle.dump(clean_data,file)
    
    final_data, target_encoder, ordinal_encoder = preprocessing(clean_data)
    with open('model/target_encoder.pkl','wb') as file:
        pickle.dump(target_encoder,file)
    with open('model/ordinal_encoder.pkl','wb') as file:
        pickle.dump(ordinal_encoder,file)


    model = trained_model(final_data) 
    with open('model/model.pkl','wb') as file:
        pickle.dump(model,file)

main()