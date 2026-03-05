"""
Script for reading and pre-processing the data.

#


"""

import pandas as pd
import numpy as np
import wfdb
import ast
import ipdb
from scipy import signal
from scipy.ndimage import median_filter
from sklearn.model_selection import train_test_split

import h5py

class DataReader(object):
    def __init__(self, path,sampling_rate=100,leads = ["I", "II", "III", "AVL", "AVR", "AVF", "V1","V2","V3","V4","V5","V6"],clss = ['STTC', 'NORM', 'MI', 'HYP', 'CD'],timesteps=1000,from_file=None, unique_clss = True):

        self.path = path
        self.lead_labels = ["I", "II", "III", "AVL", "AVR", "AVF", "V1","V2","V3","V4","V5","V6"]
        self.cls_lbls_all = ['STTC', 'NORM', 'MI', 'HYP', 'CD']
        self.sel_leads = leads
        self.leads_idx = [self.lead_labels.index(i) for i in self.sel_leads]
        self.sampling_rate = sampling_rate
        self.unique_clss = unique_clss
        # Filtering params
        self.ms_flt_array = [0.2,0.6]
        self.mfa = np.zeros(len(self.ms_flt_array), dtype='int')
        self.mean =0
        self.std =0
        
        for i in range(0, len(self.ms_flt_array)):
            self.mfa[i] = self._get_median_filter_width(self.sampling_rate,self.ms_flt_array[i])

        if from_file==None:
            self._process_data_from_raw()
        else:
            self._load(from_file)
            # Load scp_statements.csv for diagnostic aggregation
            self.agg_df = pd.read_csv(self.path+'scp_statements.csv', index_col=0)
            self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]
            self.clss_lbls = self.agg_df.diagnostic_class.unique().tolist()
        
        # Selecting data
        self.X = self.X_.copy()
        self.Y = self.Y_.copy()
        self.context = self.context_.copy()

        idx = [np.argwhere(self.Y[:,self.clss_lbls.index(cls)]==1) for cls in clss]
        idx = np.concatenate(idx).squeeze()
        idx = np.unique(idx)
        if self.unique_clss:
            idx_1_label = np.argwhere(np.sum(self.Y,axis=1)==1)[:,0]
            idx = np.intersect1d(idx,idx_1_label)
            

        self.X = self.X[idx,:timesteps,:][...,self.leads_idx]
        self.Y = self.Y[idx,...]
        self.context = self.context.iloc[idx]
        self.context["old"] = (self.context["age"]>60).astype("int64")

        print("Normalizating and test/train split")
        # normalization
        self.mean = np.mean(self.X,axis=(0,1))
        self.std = np.std(self.X,axis=(0,1))
        self.X = (self.X - self.mean)/self.std
        
        # Split data into train and test

        self.X_train, self.X_test, self.y_train, self.y_test,self.context_train, self.context_test = train_test_split(self.X, self.Y,self.context, test_size=0.2,random_state=42)


        ####### REPORT #######
        print("\n"*3)
        print("-"*100)
        print("---- DATA SUMMARY ----")
        for cls in self.clss_lbls:
            print(f"Class {cls}: {self.Y[:,self.clss_lbls.index(cls)].sum()} samples")
        
        print(f"Old att (No true labels): {self.context.old.sum()}")
        print(f"Sex att (No true labels): {self.context.sex.sum()}")
        print("-"*100)
        print("\n"*3)

        return None
    
    

    def _process_data_from_raw(self):

        # Load and convert annotation data
        print("Loading context data")
        self.context_ = pd.read_csv(self.path+'ptbxl_database.csv', index_col='ecg_id')
        self.context_.scp_codes = self.context_.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        print("Loading raw data")
        self.X_ = self._load_raw_data(self.context_)
        print("Filtering raw data")
        self.X_ = self._filter_signal(self.X_,self.mfa)

        # Load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv(self.path+'scp_statements.csv', index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        print("Processing annotations")
        self.context_['diagnostic_superclass'] = self.context_.scp_codes.apply(self._aggregate_diagnostic)
        
        self.clss_lbls = self.agg_df.diagnostic_class.unique().tolist()
        
        self.Y_ = self.context_['diagnostic_superclass'].apply(self._get_one_hot).values.tolist()
        self.Y_ = np.array(self.Y_)

        return None
    
    def _load_raw_data(self,df):
        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data
    
    def _aggregate_diagnostic(self,y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def _get_median_filter_width(self,sampling_rate, duration):
        res = int( sampling_rate*duration )
        res += ((res%2) - 1) # needs to be an odd number
        return res

    def _filter_signal(self,X,mfa):
        
        X0 = X  #read orignal signal
        for mi in range(0,len(mfa)):
            X0 = median_filter(X0,[1,mfa[mi],1]) # apply median filter one by one on top of each other
        X0 = np.subtract(X,X0)  # finally subtract from orignal signal
        return X0

    def _get_one_hot(self,df):
        a = np.zeros(len(self.clss_lbls))
        for label in df:
            a[self.clss_lbls.index(label)]=1

        return a.tolist()
    

    def store(self,output="./PTB-XL.h5"):
        print(f"saving data in {output}")
        with  h5py.File(output, 'w') as h5f:
            h5f.create_dataset('X', data=self.X_)
            h5f.create_dataset('Y', data=self.Y_)
        self.context_.to_hdf(output,"context",format="f",data_columns=True)
        
        

    def _load(self,output="./PTB-XL.h5"):
        print("Reading data from file")
        with  h5py.File(output, 'r') as h5f:
            self.X_= h5f["X"][:]
            self.Y_= h5f["Y"][:]
        self.context_ = pd.read_hdf(output,"context")
        
            
def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    Also, for every column of a str type, convert it into 
    a 'bytes' str literal of length = max(len(col)).

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int) 
                    col_type = ('S%s' % maxlen, 1)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values            
    types = df.dtypes
    numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for col in df.columns]
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        # This is in case you have problems with the encoding, remove the if branch if not
        try:
            if dtype[i].str.startswith('|S'):
                z[k] = df[k].str.encode('latin').astype('S')
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype


if __name__ == "__main__":
    
    import sys
    sys.path.append("../")
    from address import *
    PATH = "/home/diego/repositorios/24_03_05_condLS/Data/PTB-XL/"
    print(f"Getting data from {PATH}")
    data = DataReader(PATH,100,clss = ['NORM', 'CD'],from_file="/home/diego/repositorios/24_03_05_condLS/Data/PTB-XL/PTB-XL.h5")
    
    #data = DataReader(PATH,100,from_file=None); data.store("./PTB-XL/PTB-XL.h5")

    