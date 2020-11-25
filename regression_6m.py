import pandas
import pickle
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import sys

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler

import constants as c
import plot_functions

# Have function to integrate mhh in each bin of the fullmassplane
def integrate_mhh(df):
    row_list = []
    for xi in xbins:
        for yi in ybins:
            row_list.append({"m_h1":xi,"m_h2":yi,"pdf":sum(df.loc[ (df["m_h1"]==xi) & (df["m_h2"]==yi),"pdf"])})
    return pandas.DataFrame(row_list)
# Integrates the fullmassplane to get slices of mhh
def integrate_fmp(df):
    row_list = []
    for mhh in mhhbins[:-1]:
        row_list.append({"m_hh":mhh,"pdf":sum(df.loc[df["m_hh"]==mhh,"pdf"])})
    return pandas.DataFrame(row_list)


NTag = 4
# Size of the signal region, in bins
# The main tunable bin size parameters
# Fit is pretty sensitive to this. Worth experimenting.
# Hypothesis: NN should do well with a very large number of bins
NxbinsInSig=25
NybinsInSig=25
# The differing bin sizes did terribly with the regression
#mhhbins = np.array([150, 250, 262, 275, 288, 302, 317, 332, 348, 365, 383, 402, 422, 443, 465, 488, 512, 
#                  537, 563, 591, 620, 651, 683, 717, 752, 789, 828, 869, 912, 957,1004])#, 1054, 1106, 1161, 
#                  1219, 1279, 1342, 1409, 1479, 1552, 1629, 1710, 1795, 1884, 1978, 2076])
mhhbins = np.linspace(200,1000,20)

# pandas df with 6 columns: one for each mass
df = pandas.read_pickle("data_"+str(NTag)+"tag_6masses.p")
print(len(df),"Events")

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


# Now make the 6D histogram
mass_names = ["m_h1", "m_h2","m_hh","m_13","m_14","m_23","m_24"]

coord_array = np.array(df[mass_names])

m_h1 = np.array(df["m_h1"])
m_h2 = np.array(df["m_h2"])
m_hh = np.array(df["m_hh"])
m_13 = np.array(df["m_13"])
m_14 = np.array(df["m_14"])
m_23 = np.array(df["m_23"])
m_24 = np.array(df["m_24"])

bins = []
masses = [m_h1,m_h2,m_hh,m_13,m_14,m_23,m_24]
nbins = [c.n_xbins, c.n_ybins, len(c.mhhbins), 5, 5, 5, 5]
for m, n in zip(masses, nbins):
    bins.append(np.linspace(min(m), max(m), n))

hist,outbins = np.histogramdd(coord_array,bins)
bin_v = np.meshgrid(*[b[:-1] for b in outbins],indexing='ij')

data_df = pandas.DataFrame()
for i in range(len(mass_names)):
    data_df[mass_names[i]] = bin_v[i].flatten()
data_df["pdf"] = hist.flatten()

# Checks if any corners of the bin are in the SR
# Assumes giving the lower left corner of the bin
mh10,mh20=120,110
def binInSR(x,y):
    xs = x + xbinSize
    ys = y + ybinSize
    return ((0.0256 > (x -mh10)**2/x**2 +(y-mh20)**2/y**2) |
            (0.0256 > (xs-mh10)**2/xs**2+(y-mh20)**2/y**2) |
            (0.0256 > (x-mh10)**2/x**2+(ys-mh20)**2/ys**2) |
            (0.0256 > (xs-mh10)**2/xs**2+(ys-mh20)**2/ys**2))
GridBins = data_df[mass_names]

# Filter out the SR bins
if NTag == 2:
    data_dfSR = data_df.loc[binInSR(data_df["m_h1"],data_df["m_h2"])]
data_df = data_df.loc[~binInSR(data_df["m_h1"],data_df["m_h2"])]
print(len(data_df),"data points")




####################
# Moving on to the ML parts
####################

training=True

ModelName = "models/model10505050_15e_25x25_poisson_6m"

# Now lets make the regression model and train
def build_model():
    model = Sequential()

    # works well enough, runs quickly
    model.add(Dense(10, input_dim=len(mass_names), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))

    model.add(Dense(1, activation='exponential')) # Poisson loss requires output > 0 always
    model.compile(loss='poisson', optimizer='adam')
    return model

# Shuffle data points, so no training biasing/validation biasing
np.random.seed(1234)
data_df = data_df.sample(frac=1).reset_index(drop=True)
X = data_df[mass_names]
Y = data_df["pdf"]

print(np.array(Y).shape)

# Scale data to make NN easier
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
pickle.dump(scaler,open("MinMaxScaler4b_6masses.p",'wb'))

if training:
    model = build_model()
    print(model.summary())
    history = model.fit(
      X_scaled, Y,
      epochs=15, validation_split = 0.1, verbose=2, shuffle=True)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["Training Set","Validation"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.savefig(ModelName+"_Loss.png")
    plt.show()

    model.save(ModelName)
else:
    model = keras.models.load_model(ModelName)
    print(model.summary())


modeldf = GridBins
modeldf["pdf"] = model.predict(scaler.transform(GridBins))

# Plot the predicted massplane
modeldffmp = integrate_mhh(modeldf)
fig = plt.figure()
ax = fig.add_subplot(111)
xmesh = np.array(modeldffmp["m_h1"]).reshape((len(xbins),len(ybins))).transpose()
ymesh = np.array(modeldffmp["m_h2"]).reshape((len(xbins),len(ybins))).transpose()
hmesh = np.array(modeldffmp["pdf"]).reshape((len(xbins),len(ybins))).transpose()
im = ax.pcolormesh(xmesh,ymesh,hmesh)
fig.colorbar(im, ax=ax)
plot_functions.plotXhh()
plt.xlabel("$m_{h1}$")
plt.ylabel("$m_{h2}$")
plt.savefig(ModelName+"_fullmassplane.png")
plt.show()


# Plot the predicted SR mhh
if NTag==2:
    modeldfSR = modeldf.loc[binInSR(modeldf["m_h1"],modeldf["m_h2"])]
    modelmhh = list(integrate_fmp(modeldfSR)["pdf"])
    datamhh = list(integrate_fmp(data_dfSR)["pdf"])

    fig,axs = plt.subplots(2,1)
    gs = gridspec.GridSpec(2, 1,height_ratios=[3,1])
    gs.update(hspace=0)

    ax = plt.subplot(gs[0])
    ax.step(mhhbins,modelmhh+[modelmhh[-1]],'r',linewidth=2,where='post')
    XData = mhhbins[:-1]+(mhhbins[1]-mhhbins[0])/2
    ax.errorbar(XData,datamhh,yerr=np.sqrt(datamhh),fmt='k.')
    ax.set_ylabel("Counts")
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.legend(["2b SR NN Regression","2b SR Data"])

    ratio = [m/d          if d>0 else 100 for m,d in zip(modelmhh,datamhh)]
    err =   [r/np.sqrt(d) if d>0 else 0   for r,d in zip(ratio,datamhh)]
    ax = plt.subplot(gs[1])
    print(ratio)
    ax.errorbar(XData,ratio,yerr=err,fmt='k.')
    ax.plot([mhhbins[0],mhhbins[-1]],[1,1],'k--',linewidth=1)
    ax.set_ylim(0.75,1.25)
    ax.set_ylim(0.9,1.1)
    ax.set_xlabel("$m_{hh}$"+" (GeV)")
    ax.set_ylabel("Pred./Data")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    plt.savefig(ModelName+"mhhSR.png")
    plt.show()
else:
    modeldfSR = modeldf.loc[binInSR(modeldf["m_h1"],modeldf["m_h2"])]
    modelmhh = list(integrate_fmp(modeldfSR)["pdf"])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(mhhbins,modelmhh+[modelmhh[-1]],'r',linewidth=2,where='post')
    ax.set_xlabel("$m_{hh}$"+" (GeV)")
    ax.set_ylabel("Counts")
    ax.legend(["4b SR NN Regression"])
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    plt.savefig(ModelName+"mhhSR.png")
    plt.show()


# Plot massplane and 3D histogram
extra_plotting=False
if extra_plotting:
    # Just plot the massplane from data
    plt.hist2d(df["m_h1"],df["m_h2"],[xbins,ybins],[[0,300],[0,300]])
    plot_functions,plotXhh()
    plt.savefig("Fullmassplane.png")
    plt.close()

    # Make a 3D histogram of m_h1,m_h2,mhh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Make custom colormap so alpha value is 0 at min of range
    cmap = plt.cm.get_cmap("Greys")
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    alpha_cmap = ListedColormap(alpha_cmap)
    colors = [x**(0.5) for x in hist.flatten()] # Nonlinear scaling to colors so nonzero regions more visible
    img = ax.scatter(xv, yv, zv, c=colors, cmap=alpha_cmap, marker='.')

    ax.set_xlabel("$m_{h1}$")
    ax.set_ylabel("$m_{h2}$")
    ax.set_zlabel("$m_{hh}$")
    plt.show()

