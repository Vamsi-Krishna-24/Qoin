import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

df = pd.read_csv("/Users/surisettivamsikrishna/Downloads/Vamsi Pc/CODES/Qoin/Model data/master_raw_aligned.csv")
df = df.dropna()

# Assume df has columns: ICICI_Close + the factors
cols = ['ICICI_Close', 'Nifty_Close', 'US_5Y', 'US_10Y', 
        'BankNifty_Close', 'WPI_Index', 'Bond_5Y', 'CPI_Index']

data = df[cols].dropna()

# Johansen test
joh = coint_johansen(data, det_order=0, k_ar_diff=1)

print("Trace stats:", joh.lr1)
print("Critical values (90/95/99%):", joh.cvt)
print("Max-eigen stats:", joh.lr2)
print("Critical values (90/95/99%):", joh.cvm)

# Cointegrating vectors (columns of joh.evec)
print("\nCointegrating vectors:\n", joh.evec)