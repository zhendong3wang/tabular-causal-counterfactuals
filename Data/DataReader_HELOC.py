"""
Script for reading and pre-processing the Adult dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ipdb

from sklearn.preprocessing import FunctionTransformer, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer


class DataReader_HELOC(object):
    def __init__(self,path):
        self.name = "HELOC_Data"
        self.path = path
        self.RANDOM_SEED = 42

        # Define feature groups
        self.context_vars = ["MSinceOldestTradeOpen", "AverageMInFile"]
        self.context_lbls = ["MOpenAbove15years", "MInFileAbove6years"]
        self.immutable_vars = ["MSinceMostRecentTradeOpen"]
        self.numeric_vars = [
            "NumSatisfactoryTrades",
            "NumTrades60Ever2DerogPubRec",
            "NumTrades90Ever2DerogPubRec",
            "PercentTradesNeverDelq",
            "MSinceMostRecentDelq",
            "MaxDelq2PublicRecLast12M",
            "MaxDelqEver",
            "NumTotalTrades",
            "NumTradesOpeninLast12M",
            "PercentInstallTrades",
            "MSinceMostRecentInqexcl7days",
            "NumInqLast6M",  # or NumInqLast6Mexcl7days
            "NetFractionRevolvingBurden",
            "NetFractionInstallBurden",
            "NumRevolvingTradesWBalance",
            "NumInstallTradesWBalance",
            "NumBank2NatlTradesWHighUtilization",
            "PercentTradesWBalance"
        ]
        self.categorical_vars = []
        self.actionable_vars = self.numeric_vars + self.categorical_vars
        # Note: features with number of counts are integers
        self.integer_vars = [    
            "NumSatisfactoryTrades",
            "NumTrades60Ever2DerogPubRec",
            "NumTrades90Ever2DerogPubRec",
            "NumTotalTrades",
            "NumTradesOpeninLast12M",
            "NumInqLast6M",
            "NumRevolvingTradesWBalance",
            "NumInstallTradesWBalance",
            "NumBank2NatlTradesWHighUtilization"
        ]

        # Note: variable matching
        """
        ExternalRiskEstimate - consolidated indicator of risk markers (equivalent of polish BIK rate) (target var)
        MSinceOldestTradeOpen - number of months that have elapsed since first trade (context var)
        AverageMInFile - average months in file (context var)
        NumSatisfactoryTrades	number of on-time trades
        NumTrades60Ever2DerogPubRec	serious delinquency indicator
        NumTrades90Ever2DerogPubRec	more severe delinquency
        PercentTradesNeverDelq	performance indicator
        MSinceMostRecentDelq	months since last delinquency
        MaxDelq2PublicRecLast12M	recent delinquency peak
        MaxDelqEver	historical delinquency peak
        NumTotalTrades	total trade count
        NumTradesOpeninLast12M	new trades
        PercentInstallTrades	loan type composition
        MSinceMostRecentInqexcl7days	months since last inquiry
        NumInqLast6M	inquiries count
        NumInqLast6Mexcl7days	adjusted inquiries
        NetFractionRevolvingBurden	revolving debt burden
        NetFractionInstallBurden	installment debt burden
        NumRevolvingTradesWBalance	active revolving accounts
        NumInstallTradesWBalance	active installment accounts
        NumBank2NatlTradesWHighUtilization	high-utilization accounts
        PercentTradesWBalance	% of trades with balances
        """

        # Note: exclude immutable_vars from experimental modelling
        self.all_features = self.context_vars + self.numeric_vars + self.categorical_vars  
        
        self.target_var = "RiskPerformance"
        self.data = pd.read_csv(path+"heloc_dataset_v1.csv")

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
        print(f"Context 'MOpenAbove15years=1': {np.sum(self.context.values[:,0])}")
        print(f"Context 'MInFileAbove6years=1': {np.sum(self.context.values[:,1])}")
        print("-"*100)
        print("\n"*3)

        return None



    def _process_data(self):
        # Drop rows with missing values
        self.processed_data = self.data.dropna()

        # Binary encode target
        self.clss_lbls = ["Bad", "Good"]
        self.Y = (self.processed_data[self.target_var] == "Good").astype(int).values
        # One-hot: shape (N, 2)
        # Column meanings: Y[:, 0] → "bad"; Y[:, 1] → "good"
        self.diag = np.eye(len(np.unique(self.Y)))
        self.Y = self.diag[self.Y,:]

        # Context encoding
        # self.context_vars = ["MSinceOldestTradeOpen", "AverageMInFile"]
        # self.context_lbls = ["MOpenAbove15years", "MInFileAbove6years"]

        self.context = self.processed_data[self.context_vars].copy()
        self.context["MOpenAbove15years"] = (self.context["MSinceOldestTradeOpen"] > 15*12).astype(int)
        self.context["MInFileAbove6years"] = (self.context["AverageMInFile"] > 6*12).astype(int)
        # Drop original Age column if using thresholded version
        self.context = self.context.drop(columns=["MSinceOldestTradeOpen", "AverageMInFile"])
        print(pd.value_counts(self.context["MOpenAbove15years"]))
        print(pd.value_counts(self.context["MInFileAbove6years"]))

        # Features for modeling: actionable (exclude context and immutable)
        self.features_lbls = self.actionable_vars
        self.X = self.processed_data[self.features_lbls].copy()

        # Note: downsampling; original label counts
        POS_CLASS = self.clss_lbls.index("Good")
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

    PATH = "./Data/HELOC/"
    print(f"Getting data from {PATH}")

    data = DataReader_HELOC(PATH)