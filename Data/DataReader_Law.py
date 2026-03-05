"""
Script for reading and pre-processing the dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ipdb

from sklearn.preprocessing import FunctionTransformer, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer


class DataReader_Law(object):
    def __init__(self,path):
        self.name = "Law_Data"
        self.path = path
        self.RANDOM_SEED = 42

        # Define feature groups
        self.context_vars = ["male", "racetxt"]
        self.context_lbls = ["Male", "Race"]
        # self.immutable_vars = ["age", "nativecountry", "relationship"]
        self.numeric_vars = [
            "lsat",
            "ugpa",
            "zfygpa",
            "zgpa",
            "faminc",
            "decile1b",
            "decile3",
            "fulltime"
        ]
        self.categorical_vars =["tier"]
        self.actionable_vars = self.numeric_vars + self.categorical_vars
        self.integer_vars = ["fulltime"] + self.categorical_vars
        
        # Note: variable matching
        """
        @attribute decile1b real
        @attribute decile3 real
        @attribute lsat real
        @attribute ugpa real
        @attribute zfygpa real
        @attribute zgpa real
        @attribute fulltime real
        @attribute fam_inc real
        @attribute male real
        @attribute racetxt {0, 1}
        @attribute tier {1, 2, 3, 4, 5, 6}
        @attribute pass_bar {0, 1}
        """

        self.all_features = self.context_vars + self.numeric_vars + self.categorical_vars  # Note: exclude immutable_vars from experimental modelling
        self.target_var = "passbar"

        self.data = pd.read_csv(path+"law_dataset.csv")

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
        print(f"Context 'sex=Male': {np.sum(self.context.values[:,0])}")
        print(f"Context 'race=White': {np.sum(self.context.values[:,1])}")
        print("-"*100)
        print("\n"*3)

        return None



    def _process_data(self):
        # Drop rows with missing values
        self.processed_data = self.data.dropna()

        # Binary encode target
        self.clss_lbls = ["fail", "pass"]
        self.Y = self.processed_data[self.target_var].astype(int).values
        # One-hot: shape (N, 2)
        # Column meanings: Y[:, 0] → "fail"; Y[:, 1] → "pass"
        self.diag = np.eye(len(np.unique(self.Y)))
        self.Y = self.diag[self.Y,:]

        # Context encoding
        self.context = self.processed_data[self.context_vars].copy()
        self.context = self.context.astype(int)
        print(pd.value_counts(self.context["male"]))
        print(pd.value_counts(self.context["racetxt"]))
        # self.context = self.context.values

        # Features for modeling: actionable (exclude context and immutable)
        self.features_lbls = self.actionable_vars
        self.X = self.processed_data[self.features_lbls].copy()

        # Note: downsampling; original label counts
        POS_CLASS = self.clss_lbls.index("pass")
        idx_pos = np.argwhere(self.Y[:, POS_CLASS] == 1)[:, 0]
        idx_neg = np.argwhere(self.Y[:, POS_CLASS] == 0)[:, 0]
        
        # Note: len(idx_pos) = 16856; len(idx_neg) = 1836
        # print(len(idx_pos), len(idx_neg))
        np.random.seed(self.RANDOM_SEED)
        # idx_neg_sub = np.random.choice(idx_neg, len(idx_pos), replace=False)
        # idx_sub = np.hstack([idx_pos, idx_neg_sub])
        idx_pos_sub = np.random.choice(idx_pos, len(idx_neg), replace=False)
        idx_sub = np.hstack([idx_pos_sub, idx_neg])
        
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

    PATH = "./Data/Law/"
    print(f"Getting data from {PATH}")

    data = DataReader_Law(PATH)