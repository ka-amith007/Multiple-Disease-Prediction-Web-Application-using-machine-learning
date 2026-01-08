import joblib
import pandas as pd


class DiseaseModel:

    def __init__(self):
        self.all_symptoms = None
        self.symptoms = None
        self.pred_disease = None
        self.model = None
        self.diseases = None

        # Load symptom feature list from the same dataset used at training time.
        # This keeps feature ordering consistent across Windows/Linux.
        df = pd.read_csv('data/clean_dataset.tsv', sep='\t')
        self.all_symptoms = df.columns[:-1]

    def load_model(self, model_path: str):
        """Load the general disease model (scikit-learn) from a joblib file."""
        self.model = joblib.load(model_path)
        self.diseases = list(getattr(self.model, 'classes_', []))

    def predict(self, X):
        if self.model is None:
            raise RuntimeError('General disease model is not loaded')

        self.symptoms = X
        disease_pred = self.model.predict(self.symptoms)[0]
        self.pred_disease = str(disease_pred)

        disease_probability_array = self.model.predict_proba(self.symptoms)
        class_index = list(self.model.classes_).index(disease_pred)
        disease_probability = float(disease_probability_array[0, class_index])
        return self.pred_disease, disease_probability

    
    def describe_disease(self, disease_name):

        if self.diseases is not None and disease_name not in self.diseases:
            return "That disease is not contemplated in this model"
        
        # Read disease dataframe
        desc_df = pd.read_csv('data/symptom_Description.csv')
        desc_df = desc_df.apply(lambda col: col.str.strip())

        return desc_df[desc_df['Disease'] == disease_name]['Description'].values[0]

    def describe_predicted_disease(self):

        if self.pred_disease is None:
            return "No predicted disease yet"

        return self.describe_disease(self.pred_disease)
    
    def disease_precautions(self, disease_name):

        if self.diseases is not None and disease_name not in self.diseases:
            return "That disease is not contemplated in this model"

        # Read precautions dataframe
        prec_df = pd.read_csv('data/symptom_precaution.csv')
        prec_df = prec_df.apply(lambda col: col.str.strip())

        return prec_df[prec_df['Disease'] == disease_name].filter(regex='Precaution').values.tolist()[0]

    def predicted_disease_precautions(self):

        if self.pred_disease is None:
            return "No predicted disease yet"

        return self.disease_precautions(self.pred_disease)

    def disease_list(self):

        df = pd.read_csv('data/clean_dataset.tsv', sep='\t')
        y_data = df.iloc[:, -1].astype('category')
        return list(y_data.cat.categories)