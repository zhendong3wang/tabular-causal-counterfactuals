"""
Script for reading and pre-processing the dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ipdb

from sklearn.preprocessing import FunctionTransformer, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer


class DataReader_GermanCredit(object):
    def __init__(self,path):
        self.name = "GermanCredit_Data"
        self.path = path
        self.RANDOM_SEED = 42

        # Define feature groups
        self.context_vars = ["Sex", "Age"]
        self.context_lbls = ["Sex", "AgeAbove40"]
        self.immutable_vars = ["Purpose"]
        self.numeric_vars = [
            "CreditAmount",
            "Duration"
        ]
        self.categorical_vars = [
            "CheckingAccount",
            "Job",
            "Housing",
            "SavingAccounts"
        ]
        self.actionable_vars = self.numeric_vars + self.categorical_vars
        self.integer_vars = self.numeric_vars + self.categorical_vars # Note: all numeric_vars are integers
        
        # Note: variable matching
        """
        Age (numeric)
        Sex (text: male, female)
        Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
        Housing (text: own, rent, or free)
        Saving accounts (text - little, moderate, quite rich, rich)
        Checking account (numeric, in DM - Deutsch Mark)
        Credit amount (numeric, in DM)
        Duration (numeric, in month)
        Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)
        """

        # Note: exclude immutable_vars from experimental modelling
        self.all_features = self.context_vars + self.numeric_vars + self.categorical_vars  
        self.target_var = "LoanOutcome"

        self.data = pd.read_csv(path+"german_credit_data.csv")

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
        print(f"Positive class: {np.sum(self.Y[:, 1])}")
        print(f"Context 'Sex=1': {np.sum(self.context.values[:,0])}")
        print(f"Context 'AgeAbove40=1': {np.sum(self.context.values[:,1])}")
        print("-"*100)
        print("\n"*3)

        return None



    def _process_data(self):
        # Drop rows with missing values
        self.processed_data = self.data.dropna()

        # Binary encode target
        self.clss_lbls = ["bad", "good"]
        self.Y = (self.processed_data["LoanOutcome"] == "good").astype(int).values
        # One-hot: shape (N, 2)
        # Column meanings: Y[:, 0] → "bad"; Y[:, 1] → "good"
        self.diag = np.eye(len(np.unique(self.Y)))
        self.Y = self.diag[self.Y,:]

        # Context encoding
        self.context = self.processed_data[self.context_vars].copy()
        self.context["Sex"] = (self.context["Sex"] == "male").astype(int)
        self.context["AgeAbove40"] = (self.context["Age"] > 40).astype(int)
        # Drop original Age column if using thresholded version
        self.context = self.context.drop(columns=["Age"])
        print(pd.value_counts(self.context["Sex"]))
        print(pd.value_counts(self.context["AgeAbove40"]))

        # Features for modeling: actionable (exclude context and immutable)
        self.features_lbls = self.actionable_vars
        self.X = self.processed_data[self.features_lbls].copy()

        # Note: downsampling; original label counts
        POS_CLASS = self.clss_lbls.index("good")
        idx_pos = np.argwhere(self.Y[:, POS_CLASS] == 1)[:, 0]
        idx_neg = np.argwhere(self.Y[:, POS_CLASS] == 0)[:, 0]
        
        # print(len(idx_pos), len(idx_neg))
        np.random.seed(self.RANDOM_SEED)
        if len(idx_pos) < len(idx_neg):
            idx_neg_sub = np.random.choice(idx_neg, len(idx_pos), replace=False)
            idx_sub = np.hstack([idx_pos, idx_neg_sub])
        elif len(idx_pos) > len(idx_neg):
            idx_pos_sub = np.random.choice(idx_pos, len(idx_neg), replace=False)
            idx_sub = np.hstack([idx_pos_sub, idx_neg])
        else:
            print(f"len(idx_pos) = len(idx_neg) = {len(idx_pos)}")
        
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

    PATH = "./Data/GermanCredit/"
    print(f"Getting data from {PATH}")

    data = DataReader_GermanCredit(PATH)