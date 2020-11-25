import uproot
import pickle
import numpy as np

read_list = [
    "m_h1",
    "m_h2",
    "m_hh",
    "ntag",
    "pT_h1_j1",
    "pT_h1_j2",
    "pT_h2_j1",
    "pT_h2_j2",
    "eta_h1_j1",
    "eta_h1_j2",
    "eta_h2_j1",
    "eta_h2_j2",
    "phi_h1_j1",
    "phi_h1_j2",
    "phi_h2_j1",
    "phi_h2_j2"
    ]


data_filename = "data17_NN_100_bootstraps_IQR.root"
#df_master = uproot.open(data_filename)["fullmassplane"]
#for k in sorted(df_master.keys()):
#    print(k)
#exit()

# all 4b and 2b events 
df_master = uproot.open(data_filename)["fullmassplane"].pandas.df(read_list)
df_master[["m_h1","m_h2","m_hh", "ntag"]].to_pickle("data_2_and_4tag.p")