from .DataReader import DataReader
from .DataReader_GivmeCred import DataReader_GivmeCred
from .DataReader_Adult import DataReader_Adult
from .DataReader_Law import DataReader_Law
from .DataReader_GermanCredit import DataReader_GermanCredit
from .getDataReader import getData


dataReaders = {
    "ECG":DataReader,
    "GIVECREDIT":DataReader_GivmeCred, 
    "ADULT": DataReader_Adult, 
    "LAW": DataReader_Law,
    "GERMANCREDIT": DataReader_GermanCredit
}