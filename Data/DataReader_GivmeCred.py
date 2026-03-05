"""
Script for reading and pre-processing the Give Me Some Credit Data
"""

"""
Feature Name                            | Type              | Description
----------------------------------------|--------------------|---------------------------------------------------------------
SeriousDlqin2yrs                        | Binary             | Target variable. 1 if the person had a serious delinquency
                                        |                    | (90+ days overdue) in the next 2 years, 0 otherwise.

RevolvingUtilizationOfUnsecuredLines    | Continuous Numeric | Ratio of total balance on unsecured credit lines (e.g., credit
                                        |                    | cards) to total credit limit. Usually between 0 and 1, but can exceed 1.

age                                     | Integer Numeric    | Age of the individual in years.

NumberOfTime30-59DaysPastDueNotWorse    | Integer Numeric    | Number of times the person was 30–59 days past due, but not worse,
                                        |                    | in the past 2 years.

DebtRatio                               | Continuous Numeric | Ratio of monthly debt payments to gross monthly income.

MonthlyIncome                           | Continuous Numeric | Self-reported monthly income. May contain missing values.

NumberOfOpenCreditLinesAndLoans         | Integer Numeric    | Number of open credit lines (credit cards, loans, etc.).

NumberOfTimes90DaysLate                 | Integer Numeric    | Number of times the person has been more than 90 days overdue.

NumberRealEstateLoansOrLines            | Integer Numeric    | Number of real estate loans or credit lines (e.g., mortgages).

NumberOfTime60-89DaysPastDueNotWorse    | Integer Numeric    | Number of times the person was 60–89 days past due, but not worse,
                                        |                    | in the past 2 years.

NumberOfDependents                      | Integer Numeric    | Number of dependents (e.g., children). May contain missing values.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ipdb

from sklearn.preprocessing import FunctionTransformer, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer


class DataReader_GivmeCred(object):
    def __init__(self,path):
        self.name = "GivmeCred_Data"
        self.path = path
        self.log_vars = [
            # 'NumberOfTime30-59DaysPastDueNotWorse',
            # 'NumberOfTime60-89DaysPastDueNotWorse',                    
            'NumberOfTime3059DaysPastDueNotWorse',
            'NumberOfTime6089DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberRealEstateLoansOrLines'
        ]

        self.quantile_vars = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']
        self.standard_vars = [ 'MonthlyIncome']

        self.integer_vars    = [
            # 'NumberOfTime30-59DaysPastDueNotWorse',
            # 'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTime3059DaysPastDueNotWorse',
            'NumberOfTime6089DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberRealEstateLoansOrLines',
            'RevolvingUtilizationOfUnsecuredLines',
            'MonthlyIncome'
            ]


        self.continuous_vars = ['DebtRatio']


        self.data = pd.read_csv(path+"cs-training.csv")

        self._process_data()


        
        self.X_train, self.X_test, self.y_train, self.y_test,self.context_train, self.context_test = train_test_split(self.X, self.Y,self.context, test_size=0.2,random_state=42)

        # Normalization
        
        # self.mean = self.X_train.mean(axis=0)
        # self.std = self.X_train.std(axis=0)

        # self.X_train = (self.X_train - self.mean)/self.std
        # self.X_test = (self.X_test - self.mean)/self.std

        self.X_train = self.scaler_fit_transform(self.X_train)
        self.X_test = self.scaler_transform(self.X_test)

        # ####### REPORT #######
        print("\n"*3)
        print("-"*100)
        print("---- DATA SUMMARY ----")
        for i,cls in enumerate(self.clss_lbls):
            print(f"Class {cls}: {self.Y[:,i].sum()} samples")
        
        print(f"Age above 50 att: {np.sum(self.context.values[:,0])}")
        print(f"Dependents more than one: {np.sum(self.context.values[:,1])}")
        print("-"*100)
        print("\n"*3)

        return None



    def _process_data(self):
        epsilon = 1e-4
        self.processed_data = self.data.dropna()
        self.processed_data = self.processed_data.drop(columns='Unnamed: 0')
        self.processed_data = self.processed_data + 0
        self.processed_data = self.processed_data.loc[self.processed_data['age']<88]

        self.clss_lbls = ["Good Standing","Default Risk"]
        self.context_lbls_ =["age","NumberOfDependents"]

        self.target_var = "SeriousDlqin2yrs"
        self.Y =  self.processed_data[self.target_var].values
        self.diag = np.eye(len(np.unique(self.Y)))
        self.Y = self.diag[self.Y,:]

        # Subsampling the majority class # note: reduced the sample size
        idx_no_risk = np.argwhere(self.Y[:,0] == 1)[:,0]
        idx_risk = np.argwhere(self.Y[:,0] == 0)[:,0]
        idx_no_risk_sub = np.random.choice(idx_no_risk,len(idx_risk),replace=False)
        idx_sub = np.hstack([idx_risk,idx_no_risk_sub])
    
        self.Y = self.Y[idx_sub]
        self.processed_data = self.processed_data.iloc[idx_sub]
        
        # The feature names order of GiveMeCredit
        """
        # 1. RevolvingUtilizationOfUnsecuredLines
        # 2. NumberOfTime30-59DaysPastDueNotWorse
        # 3. DebtRatio
        # 4. MonthlyIncome
        # 5. NumberOfOpenCreditLinesAndLoans
        # 6. NumberOfTimes90DaysLate
        # 7. NumberRealEstateLoansOrLines
        # 8. NumberOfTime60-89DaysPastDueNotWorse
        """
        self.features_lbls = [i for i in self.processed_data.columns[1:] if i not in self.context_lbls_]
        self.X = self.processed_data[self.features_lbls].values

        
        self.context = self.processed_data[self.context_lbls_].values

        self.context[:,0] =  self.context[:,0]>50
        self.context[:,1] =  self.context[:,1]>0

        self.context = pd.DataFrame(self.context.astype(int),columns=["ageAbove50","Dependents"])
        

    def scaler_fit_transform(self,X):
        """
        Transform X taking into the account the nature of the feats in the dataset
        INPUT
            X[numpy.narray]

        OUTPUT
            X_scaled
        """

        x_df = pd.DataFrame(data=X,columns = self.features_lbls)


        # Log
        X_log = x_df[self.log_vars].copy()
        log_imputer = SimpleImputer(strategy='constant', fill_value=0).fit(X_log)
        X_log_imp = log_imputer.transform(X_log)
        X_log_trans = np.log1p(X_log_imp)
        self.log_scaler = StandardScaler().fit(X_log_trans)
        X_log_scaled = self.log_scaler.transform(X_log_trans)

        # Quantile
        X_quant = x_df[self.quantile_vars].copy()
        quant_imputer = SimpleImputer(strategy='constant', fill_value=0).fit(X_quant)
        X_quant_imp = quant_imputer.transform(X_quant)
        self.quant_scaler = QuantileTransformer(output_distribution='normal').fit(X_quant_imp)
        X_quant_scaled = self.quant_scaler.transform(X_quant_imp)

        # Standard
        X_std = x_df[self.standard_vars].copy()
        std_imputer = SimpleImputer(strategy='median').fit(X_std)
        X_std_imp = std_imputer.transform(X_std)
        self.std_scaler = StandardScaler().fit(X_std_imp)
        X_std_scaled = self.std_scaler.transform(X_std_imp)

        
        X_scaled = np.hstack([X_log_scaled, X_quant_scaled, X_std_scaled])
        X_scaled = pd.DataFrame(data=X_scaled,columns = self.log_vars + self.quantile_vars + self.standard_vars)

        X_scaled = X_scaled[self.features_lbls]

        return X_scaled.values


    def scaler_transform(self,X):
        """
        Transform X taking into the account the nature of the feats in the dataset
        INPUT
            X[numpy.narray]

        OUTPUT
            X_scaled
        """

        x_df = pd.DataFrame(data=X,columns = self.features_lbls)


        # Log
        X_log = x_df[self.log_vars].copy()
        log_imputer = SimpleImputer(strategy='constant', fill_value=0).fit(X_log)
        X_log_imp = log_imputer.transform(X_log)
        X_log_trans = np.log1p(X_log_imp)
        X_log_scaled = self.log_scaler.transform(X_log_trans)

        # Quantile
        X_quant = x_df[self.quantile_vars].copy()
        quant_imputer = SimpleImputer(strategy='constant', fill_value=0).fit(X_quant)
        X_quant_imp = quant_imputer.transform(X_quant)
        X_quant_scaled = self.quant_scaler.transform(X_quant_imp)

        # Standard
        X_std = x_df[self.standard_vars].copy()
        std_imputer = SimpleImputer(strategy='median').fit(X_std)
        X_std_imp = std_imputer.transform(X_std)
        X_std_scaled = self.std_scaler.transform(X_std_imp)

        X_scaled = np.hstack([X_log_scaled, X_quant_scaled, X_std_scaled])
        X_scaled = pd.DataFrame(data=X_scaled,columns = self.log_vars + self.quantile_vars + self.standard_vars)

        X_scaled = X_scaled[self.features_lbls]

        return X_scaled.values


    def scaler_inverse_transform(self,X):
        """
        Unscale X taking into the previous normalization and the nature of the feats in the dataset
        INPUT
            X[numpy.narray]

        OUTPUT
            X_unscaled
        """
        x_df = pd.DataFrame(data=X,columns = self.features_lbls)
        X_log = x_df[self.log_vars].copy()
        X_quant = x_df[self.quantile_vars].copy()
        X_std = x_df[self.standard_vars].copy()

        X_log_unscaled = self.log_scaler.inverse_transform(X_log)
        X_log_unlog = np.expm1(X_log_unscaled)

        X_quant_unscaled = self.quant_scaler.inverse_transform(X_quant)

        X_std_unscaled = self.std_scaler.inverse_transform(X_std)

        X_unscaled = pd.DataFrame(
            np.hstack([X_log_unlog, X_quant_unscaled, X_std_unscaled]),
            columns=self.log_vars + self.quantile_vars + self.standard_vars
        )

        X_unscaled = X_unscaled[self.features_lbls]

        for var in self.integer_vars:
            # Note: astype(int) only rounds down, e.g. int(4.8)=4; while np.around(int(4.8))=5
            X_unscaled[var] = np.around(X_unscaled[var].values).astype(int)

        return X_unscaled.values

if __name__=="__main__":

    import matplotlib.pyplot as plt

    plt.ion()

    PATH = "/home/diego/repositorios/24_03_05_condLS/Data/GiveMeCredit/"
    print(f"Getting data from {PATH}")

    data = DataReader_GivmeCred(PATH)