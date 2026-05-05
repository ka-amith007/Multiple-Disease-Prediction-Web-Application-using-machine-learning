import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.naive_bayes import BernoulliNB

# Machine learning model: scikit-learn (Bernoulli Naive Bayes)
# This avoids native-compiled runtime dependencies like xgboost and is
# reliable on Streamlit Community Cloud.

# import the dataset
dataset_df = pd.read_csv('data/dataset.csv')

# Preprocess
dataset_df = dataset_df.apply(lambda col: col.str.strip())

test = pd.get_dummies(dataset_df.filter(regex='Symptom'), prefix='', prefix_sep='')
test = test.groupby(test.columns, axis=1).agg(np.max)
clean_df = pd.merge(test,dataset_df['Disease'], left_index=True, right_index=True)

clean_df.to_csv('data/clean_dataset.tsv', sep='\t', index=False)

# Preprocessing
X_data = clean_df.iloc[:,:-1]
y_data = clean_df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(
	X_data,
	y_data,
	test_size=0.2,
	random_state=42,
	stratify=y_data,
)

# Init classifier
model = BernoulliNB()

# Fit
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Test accuracy
print(f"The accuracy of the model is {accuracy_score(y_test, preds)}")

# Export model
joblib.dump(model, 'model/disease_model.joblib')
