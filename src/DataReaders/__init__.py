from .DataReader_GiveMeCredit import DataReader_GiveMeCredit
from .DataReader_Adult import DataReader_Adult
from .DataReader_Law import DataReader_Law
from .DataReader_GermanCredit import DataReader_GermanCredit
from .DataReader_HELOC import DataReader_HELOC
from .getDataReader import getData

dataReaders = {
    "GIVEMECREDIT":DataReader_GiveMeCredit,
    "ADULT": DataReader_Adult, 
    "LAW": DataReader_Law,
    "GERMANCREDIT": DataReader_GermanCredit,
    "HELOC": DataReader_HELOC
}