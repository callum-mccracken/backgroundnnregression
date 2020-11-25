import pandas
import pickle
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import sys
import os
from tqdm import tqdm
import binning
import plot_functions

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler

# Have function to integrate mhh in each bin of the fullmassplane
def integrate_mhh(df):
    row_list = []
    for xi in tqdm(xbins):
        for yi in ybins:
            row_list.append({"mh1":xi,"mh2":yi,"pdf":sum(df.loc[ (df["mh1"]==xi) & (df["mh2"]==yi),"pdf"])})
    return pandas.DataFrame(row_list)
# Integrates the fullmassplane to get slices of mhh
def integrate_fmp(df):
    row_list = []
    for mhh in mhhbins[:-1]:
        row_list.append({"mhh":mhh,"pdf":sum(df.loc[df["mhh"]==mhh,"pdf"])})
    return pandas.DataFrame(row_list)

epochs = 30
training = True
n_mhhbins = 50

NxbinsInSig = 25
NybinsInSig = 25 
mhhbins = np.linspace(200,1000,n_mhhbins)

# pandas df with 3 columns: m_h1, m_h2, and m_hh
df2 = pandas.read_pickle("data_2tag.p")
df4 = pandas.read_pickle("data_4tag.p")
print(len(df2) + len(df4),"Events")

""" This is just a bunch of stuff to ensure that no bins are partly in the
signal region. This is necessary so the fit isn't biased right at the edge of the SR.
That would be terrible."""

# First, make histogram with signal region box:
# The exact edges of our signal region, computed analytically
sxmin,sxmax=103.45,142.86
symin,symax=94.82,130.95

# Approximate min/max values in the massplane. Chooses closest bin edge to this boundary
# Based on extent of the background. Might be able to tune a little
xmin,xmax = 50,250 
ymin,ymax = 40,200

xbinSize = (sxmax-sxmin)/NxbinsInSig
ybinSize = (symax-symin)/NybinsInSig
xbins = np.arange(sxmin-int((sxmin-xmin)/xbinSize)*xbinSize,int(xmax/xbinSize)*xbinSize,xbinSize)
ybins = np.arange(symin-int((symin-ymin)/ybinSize)*ybinSize,int(ymax/ybinSize)*ybinSize,ybinSize)

# Now make the 3D histogram
coord_array_2b = np.array(df2[["m_h1","m_h2","m_hh"]])
coord_array_4b = np.array(df4[["m_h1","m_h2","m_hh"]])

hist3d_2b,[xbins,ybins,mhhbins] = np.histogramdd(coord_array_2b,[xbins,ybins,mhhbins])
hist3d_4b, _ = np.histogramdd(coord_array_4b,[xbins,ybins,mhhbins])
xv,yv,zv = np.meshgrid(xbins[:-1],ybins[:-1],mhhbins[:-1],indexing='ij')

assert len(hist3d_2b.flatten()) == len(hist3d_4b.flatten())

assert len(xv.flatten()) == len(yv.flatten()) == len(zv.flatten()) == len(hist3d_4b.flatten())

n_hist_points = len(xv.flatten())

# make dataframe
data_df = pandas.DataFrame()
data_df["mh1"] = np.empty(n_hist_points*2)
data_df["mh2"] = np.empty(n_hist_points*2)
data_df["mhh"] = np.empty(n_hist_points*2)
data_df["pdf"] = np.empty(n_hist_points*2)
data_df["ntag"] = np.empty(n_hist_points*2)

# add 2b data
data_df["mh1"][:n_hist_points] = xv.flatten()
data_df["mh2"][:n_hist_points] = yv.flatten()
data_df["mhh"][:n_hist_points] = zv.flatten()
data_df["pdf"][:n_hist_points] = hist3d_2b.flatten()
data_df["ntag"][:n_hist_points] = 2

# then add on the 4b data
data_df["mh1"][n_hist_points:] = xv.flatten()
data_df["mh2"][n_hist_points:] = yv.flatten()
data_df["mhh"][n_hist_points:] = zv.flatten()
data_df["pdf"][n_hist_points:] = hist3d_4b.flatten()
data_df["ntag"][n_hist_points:] = 4

print(len(data_df))

# Checks if any corners of the bin are in the SR
# Assumes giving the lower left corner of the bin
GridBins = data_df[["mh1","mh2","mhh",'ntag']]

# SR bins with ntag = 2
data_dfSR = data_df.loc[binning.binInSR(data_df["mh1"],data_df["mh2"]) & data_df["ntag"] == 2]
# all other bins
data_df = data_df.loc[~binning.binInSR(data_df["mh1"],data_df["mh2"])]
print(len(data_df),"data points")

####################
# Moving on to the ML parts
####################

layers = [10, 50, 50, 50]
layers_str = "".join([str(l) for l in layers])
ModelName = f"models/model_2b4b_{layers_str}_{epochs}e_{NxbinsInSig}x{NybinsInSig}_poisson_{n_mhhbins}mhh"

# Now lets make the regression model and train
def build_model():
    model = Sequential()
    model.add(Dense(layers[0], input_dim=4, activation='relu'))
    for l in layers[1:]:
        model.add(Dense(l, activation='relu'))
    model.add(Dense(1, activation='exponential')) # Poisson loss requires output > 0 always
    model.compile(loss='poisson', optimizer='adam')
    print('built model')
    return model

# Shuffle data points, so no training biasing/validation biasing
np.random.seed(1234)
data_df = data_df.sample(frac=1).reset_index(drop=True)

X = data_df[["mh1","mh2","mhh","ntag"]]
Y = data_df["pdf"]

# Scale data to make NN easier
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
pickle.dump(scaler,open("MinMaxScaler2b4b.p",'wb'))
print('data ready for nn')

if training:
    model = build_model()
    print(model.summary())
    history = model.fit(
        X_scaled, Y,
        epochs=epochs, validation_split = 0.1, verbose=1, shuffle=True)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["Training Set","Validation"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.savefig(ModelName+"_Loss.png")
    plt.cla();plt.clf()
    #plt.show()
    plt.close()

    model.save(ModelName)
else:
    model = keras.models.load_model(ModelName)
    print(model.summary())
    print('model opened')


modeldf = GridBins
modeldf["pdf"] = model.predict(scaler.transform(GridBins), verbose=1)
pickle.dump(modeldf["pdf"],open(f"model_output_2b4b.p",'wb'))
print('predictions made')

# split predictions into 2b and 4b dataframes
modeldf_2b = pandas.DataFrame()
modeldf_4b = pandas.DataFrame()

# add 2b data
modeldf_2b["mh1"] = modeldf["mh1"][:n_hist_points]
modeldf_2b["mh2"] = modeldf["mh2"][:n_hist_points]
modeldf_2b["mhh"] = modeldf["mhh"][:n_hist_points]
modeldf_2b["pdf"] = modeldf["pdf"][:n_hist_points]
modeldf_2b["ntag"] = modeldf["ntag"][:n_hist_points]

# then add on the 4b data
modeldf_4b["mh1"] = modeldf["mh1"][n_hist_points:]
modeldf_4b["mh2"] = modeldf["mh2"][n_hist_points:]
modeldf_4b["mhh"] = modeldf["mhh"][n_hist_points:]
modeldf_4b["pdf"] = modeldf["pdf"][n_hist_points:]
modeldf_4b["ntag"] = modeldf["ntag"][n_hist_points:]



modeldffmp_2b = integrate_mhh(modeldf_2b)
modeldffmp_4b = integrate_mhh(modeldf_4b)
print('m_hh integrated')


# plot 2b
fig = plt.figure()
ax = fig.add_subplot(111)
xmesh = np.array(modeldffmp_2b["mh1"]).reshape((len(xbins),len(ybins))).transpose()
ymesh = np.array(modeldffmp_2b["mh2"]).reshape((len(xbins),len(ybins))).transpose()
hmesh = np.array(modeldffmp_2b["pdf"]).reshape((len(xbins),len(ybins))).transpose()
ax.pcolormesh(xmesh,ymesh,hmesh, shading='auto')
plot_functions.plotXhh()
plt.xlabel("$m_{h1}$")
plt.ylabel("$m_{h2}$")
plt.title(f"Model-predicted massplane, NTag=2")
plt.savefig(ModelName+"_predicted_2b_massplane.png")
plt.cla();plt.clf()
plt.close()

# plot 4b
fig = plt.figure()
ax = fig.add_subplot(111)
xmesh = np.array(modeldffmp_4b["mh1"]).reshape((len(xbins),len(ybins))).transpose()
ymesh = np.array(modeldffmp_4b["mh2"]).reshape((len(xbins),len(ybins))).transpose()
hmesh = np.array(modeldffmp_4b["pdf"]).reshape((len(xbins),len(ybins))).transpose()
ax.pcolormesh(xmesh,ymesh,hmesh, shading='auto')
plot_functions.plotXhh()
plt.xlabel("$m_{h1}$")
plt.ylabel("$m_{h2}$")
plt.title(f"Model-predicted massplane, NTag=4")
plt.savefig(ModelName+"_predicted_4b_massplane.png")
plt.cla();plt.clf()
plt.close()

print('plotted predicted massplanes')
