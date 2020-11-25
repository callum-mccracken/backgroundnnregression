import pickle
from scipy import stats
import sympy as sp
import matplotlib.pyplot as plt
import plot_functions
import pandas as pd
import binning
import constants as c
import numpy as np
import scipy.stats as st

# names of data files
pickle_name_2b = "data_2tag.p"
pickle_name_2b_full = "data_2tag_full.p"
pickle_name_4b = "data_4tag.p"

# get dataframes with variables m_h1, m_h2, m_hh
df_2b = pd.read_pickle(pickle_name_2b)
df_2b_full = pd.read_pickle(pickle_name_2b_full)
df_4b = pd.read_pickle(pickle_name_4b)

# separate into 3 different regions
print('making histogram')
df_all, df_SR, df_no_SR = binning.make_histogram('data_2tag.p', 2)

# plot full unsmoothed massplane
print('plotting unsmoothed massplane')
plot_functions.plot_fullmassplane_from_df(
    df_all, 'unsmoothed_massplane.png', show=True)

X = np.array(df_all[["m_h1", "m_h2", "m_hh"]]).T
df_all[["m_h1", "m_h2", "m_hh"]].to_pickle("X.p")
df_all["pdf"].to_pickle("Y.p")


# then pass off the KDE stuff to kde.py which makes new_Y.p
# TODO: make this less janky later

new_df = pd.DataFrame()
new_df[["m_h1", "m_h2", "m_hh"]] = df_all[["m_h1", "m_h2", "m_hh"]]
new_df["pdf"] = pd.read_pickle('new_Y.p')

plot_functions.plot_fullmassplane_from_df(
    new_df, 'smoothed_massplane.png', show=True)


# save as X = [m_h1, m_h2, m_hh], Y = pdf, for use in NN


