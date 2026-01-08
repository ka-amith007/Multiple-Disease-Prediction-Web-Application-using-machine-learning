import pandas as pd
import numpy as np


def prepare_symptoms_array(symptoms):
    '''
    Convert a list of symptoms to a ndim(X) (in this case 131) that matches the
    dataframe used to train the machine learning model

    Output:
    - X (np.array) = X values ready as input to ML model to get prediction
    '''
    df = pd.read_csv('data/clean_dataset.tsv', sep='\t')
    feature_cols = df.columns[:-1]
    symptoms_array = np.zeros((1, len(feature_cols)))
    
    for symptom in symptoms:
        symptom_idx = feature_cols.get_loc(symptom)
        symptoms_array[0, symptom_idx] = 1

    # Return a DataFrame so the model receives stable feature names/order.
    return pd.DataFrame(symptoms_array, columns=feature_cols)