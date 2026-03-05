"""
Script for reading and pre-processing the Adult dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ipdb

from sklearn.preprocessing import FunctionTransformer, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer


class DataReader_Adult(object):
    def __init__(self,path):
        self.name = "Adult_Data"
        self.path = path
        self.RANDOM_SEED = 42

        # Define feature groups
        self.context_vars = ["sex", "race"]  # binary
        self.context_lbls = ["Male", "White"]
        # self.context["sex"] = (self.context["sex"] == "Male").astype(int)
        # self.context["race"] = (self.context["race"] == "White").astype(int)
        self.immutable_vars = ["age", "nativecountry", "relationship"]
        self.actionable_vars = [
            "educationnum", "workclass", "occupation",
            "hoursperweek", "capitalgain", "capitalloss", "maritalstatus"
        ]
        # self.actionable_vars = self.numeric_vars + self.categorical_vars # TODO: order of features
        self.numeric_vars = [
            "educationnum", "hoursperweek", "capitalgain", "capitalloss"
        ]
        self.categorical_vars =["workclass", "occupation", "maritalstatus"]
        self.integer_vars = [
            "educationnum",
            "hoursperweek",
            "capitalgain", 
            "capitalloss", 
        ] + self.categorical_vars
        
        # Note: variable matching
        """
        "age", - immutable_vars
        "workclass", - categorical_vars 
        "fnlwgt",
        "education", 
        "education-num", - integer_vars
        "marital-status", - categorical_vars
        "occupation", - categorical_vars
        "relationship", - immutable_vars
        "race", - context_vars
        "sex", - context_vars
        "capital-gain", - integer_vars
        "capital-loss", - integer_vars
        "hours-per-week", - integer_vars
        "native-country", - immutable_vars
        "income" - target
        """

        self.all_features = self.context_vars + self.actionable_vars # Note: exclude immutable_vars from experimental modelling
        self.target_var = "income"

        self.data = pd.read_csv(path+"adult.csv")

        self._process_data()
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test, \
        self.context_train, self.context_test = train_test_split(
            self.X, self.Y, self.context, test_size=0.2, random_state=self.RANDOM_SEED, stratify=self.Y
        )

        # Scaling numeric features
        self.X_train = self.scaler_fit_transform(self.X_train)
        self.X_test = self.scaler_transform(self.X_test)

        # ####### REPORT #######
        print("\n"*3)
        print("-"*100)
        print("---- DATA SUMMARY ----")
        print(f"Total samples: {len(self.Y)}")
        print(f"Positive class (>50K): {np.sum(self.Y[:, 1])}")
        print(f"Context 'sex=Male': {np.sum(self.context.values[:,0])}")
        print(f"Context 'race=White': {np.sum(self.context.values[:,1])}")
        print("-"*100)
        print("\n"*3)

        return None



    def _process_data(self):
        # Drop rows with missing values
        self.processed_data = self.data.replace('?', np.nan).dropna()

        # Binary encode target
        self.clss_lbls = ["<=50K", ">50K"]
        self.Y = (self.processed_data[self.target_var] == ">50K").astype(int).values
        # One-hot: shape (N, 2)
        # Column meanings: Y[:, 0] → "<=50K"; Y[:, 1] → ">50K"
        self.diag = np.eye(len(np.unique(self.Y)))
        self.Y = self.diag[self.Y,:]

        # # Previous code: Context encoding: sex, race
        # self.context = self.processed_data[self.context_vars].copy().values
        # self.context[:,0] =  self.context[:,0] == "Male"
        # self.context[:,1] =  self.context[:,1] == "White"
        # self.context = pd.DataFrame(self.context.astype(int), columns=["sex", "race"])

        # Context encoding: sex, race
        self.context = self.processed_data[self.context_vars].copy()
        self.context["sex"] = (self.context["sex"] == "Male").astype(int)
        self.context["race"] = (self.context["race"] == "White").astype(int)
        print(pd.value_counts(self.context["sex"]))
        print(pd.value_counts(self.context["race"]))
        # self.context = self.context.values

        # Features for modeling: actionable (exclude context and immutable)
        self.features_lbls = self.actionable_vars
        self.X = self.processed_data[self.features_lbls].copy()

        # Note: downsampling; original label counts -- <=50K (76%); >50K (24%)
        POS_CLASS = self.clss_lbls.index(">50K")
        idx_pos = np.argwhere(self.Y[:, POS_CLASS] == 1)[:, 0]
        idx_neg = np.argwhere(self.Y[:, POS_CLASS] == 0)[:, 0]
        np.random.seed(self.RANDOM_SEED)
        idx_neg_sub = np.random.choice(idx_neg, len(idx_pos), replace=False)
        idx_sub = np.hstack([idx_pos, idx_neg_sub])
        
        # use iloc for DataFrame
        self.X = self.X.iloc[idx_sub].reset_index(drop=True)
        self.context = self.context.iloc[idx_sub]
        self.Y = self.Y[idx_sub]

        # Encode categorical actionable vars as codes
        # TODO: record the encoded codes/categories
        for col in self.categorical_vars:
            self.X[col] = self.X[col].astype('category').cat.codes

        # Keep track of feature names after one-hot
        self.features_lbls = self.X.columns.tolist()
        self.X = self.X.values

        # previous version for GiveMeCredit
        # epsilon = 1e-4
        # self.processed_data = self.data.dropna()
        # self.processed_data = self.processed_data.drop(columns='Unnamed: 0')
        # self.processed_data = self.processed_data + 0
        # self.processed_data = self.processed_data.loc[self.processed_data['age']<88]

        # self.clss_lbls = ["Good Standing","Default Risk"]
        # self.context_lbls_ =["age","NumberOfDependents"]
        # self.Y =  self.processed_data["SeriousDlqin2yrs"].values
        # self.diag = np.eye(len(np.unique(self.Y)))
        # self.Y = self.diag[self.Y,:]

        # # Subsampling the majority class # note: reduced the sample size
        # idx_no_risk = np.argwhere(self.Y[:,0] == 1)[:,0]
        # idx_risk = np.argwhere(self.Y[:,0] == 0)[:,0]
        # idx_no_risk_sub = np.random.choice(idx_no_risk,len(idx_risk),replace=False)
        # idx_sub = np.hstack([idx_risk,idx_no_risk_sub])
    
        # self.Y = self.Y[idx_sub]
        # self.processed_data = self.processed_data.iloc[idx_sub]
        # self.features_lbls = [i for i in self.processed_data.columns[1:] if i not in self.context_lbls_]
        # self.X = self.processed_data[self.features_lbls].values
        
        # self.context = self.processed_data[self.context_lbls_].values
        # self.context[:,0] =  self.context[:,0]>50
        # self.context[:,1] =  self.context[:,1]>0
        # self.context = pd.DataFrame(self.context.astype(int),columns=["ageAbove50","Dependents"])
        

    def scaler_fit_transform(self,X):
        X_df = pd.DataFrame(X, columns=self.features_lbls)
        scaler = StandardScaler()
        X_df[self.numeric_vars] = scaler.fit_transform(X_df[self.numeric_vars])
        self.scaler = scaler
        return X_df.values

    def scaler_transform(self,X):
        X_df = pd.DataFrame(X, columns=self.features_lbls)
        X_df[self.numeric_vars] = self.scaler.transform(X_df[self.numeric_vars])
        return X_df.values

    def scaler_inverse_transform(self,X):
        X_df = pd.DataFrame(X, columns=self.features_lbls)
        X_df[self.numeric_vars] = self.scaler.inverse_transform(X_df[self.numeric_vars])

        for var in self.integer_vars:
            X_df[var] = np.around(X_df[var].values).astype(int)
        return X_df.values

if __name__=="__main__":

    # import matplotlib.pyplot as plt
    # plt.ion()

    PATH = "./Data/Adult/"
    print(f"Getting data from {PATH}")

    data = DataReader_Adult(PATH)