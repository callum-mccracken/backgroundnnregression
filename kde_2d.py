import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random
import pickle
from matplotlib.colors import ListedColormap
import plot_functions
import constants as c
from tqdm import tqdm
import os

NTag = 4
print('running with NTag =', NTag)


def binInSR(x,y):
    """
    Checks if any corners of the bin are in the SR
    Assumes giving the lower left corner of the bin
    """
    xs = x + c.xbinSize
    ys = y + c.ybinSize
    return ((0.0256 > (x - c.m_h1_0)**2 / x**2 + (y - c.m_h2_0)**2 / y**2) |
            (0.0256 > (xs - c.m_h1_0)**2 / xs**2 + (y - c.m_h2_0)**2 / y**2) |
            (0.0256 > (x - c.m_h1_0)**2 / x**2 + (ys - c.m_h2_0)**2 / ys**2) |
            (0.0256 > (xs - c.m_h1_0)**2 / xs**2 + (ys - c.m_h2_0)**2 / ys**2))

def integrate_mhh(df):
    """
    Gets sum of m_hh in each bin
    """
    row_list = []
    for xi in tqdm(c.xbins):
        for yi in c.ybins:
            row_list.append({"m_h1":xi,"m_h2":yi,"pdf":sum(df.loc[ (df["m_h1"]==xi) & (df["m_h2"]==yi),"pdf"])})
    return pd.DataFrame(row_list)

df = pd.read_pickle(f"data_{NTag}tag.p")
# Now make the 3D histogram
coord_array = np.array(df[["m_h1","m_h2","m_hh"]])

hist3d,[xbins,ybins,mhhbins] = np.histogramdd(coord_array,[c.xbins,c.ybins,c.mhhbins])
xv,yv,zv = np.meshgrid(xbins[:-1],ybins[:-1],mhhbins[:-1],indexing='ij')

data_df = pd.DataFrame()
data_df["m_h1"] = xv.flatten()
data_df["m_h2"] = yv.flatten()
data_df["m_hh"] = zv.flatten()
data_df["pdf"] = hist3d.flatten()

pickle.dump(np.array(data_df[["m_h1","m_h2","m_hh"]]), open("3mnn_X.p", 'wb'))
pickle.dump(np.array(data_df["pdf"]), open("3mnn_Y.p", 'wb'))

# Filter out the SR bins if needed
if NTag == 4:
    data_df = data_df.loc[~binInSR(data_df["m_h1"],data_df["m_h2"])]

# bin into histogram, save
fmp = integrate_mhh(data_df)
pickle.dump(fmp, open(f"fmp_{NTag}b.p", 'wb'))


# plot initial massplane
fig = plt.figure()
ax = fig.add_subplot(111)
xmesh = np.array(np.array(fmp["m_h1"]).reshape((len(c.xbins),len(c.ybins))).T)
ymesh = np.array(np.array(fmp["m_h2"]).reshape((len(c.xbins),len(c.ybins))).T)
hmesh = np.array(np.array(fmp["pdf"]).reshape((len(c.xbins),len(c.ybins))).T)
ax.pcolormesh(xmesh,ymesh,hmesh)
plot_functions.plotXhh()
plt.xlabel("$m_{h1}$")
plt.ylabel("$m_{h2}$")
plt.savefig(f"kde_2d_fullmassplane_{NTag}b_init.png")

# do kde to get new density
if NTag == 2:
    x = xmesh.flatten()
    y = ymesh.flatten()
    h = hmesh.flatten()
elif NTag == 4:
    mask_indices = ~binInSR(xmesh,ymesh)
    x = xmesh[mask_indices].flatten()
    y = ymesh[mask_indices].flatten()
    h = hmesh[mask_indices].flatten()

print('doing kde without signal region,', len(x), 'pts instead of', len(xmesh.flatten()))
kde = stats.gaussian_kde([x, y], weights=h)#, bw_method=0.02)
print(kde.factor)

# calculate kde density
new_density = kde([data_df["m_h1"], data_df["m_h2"]])
print('done the long part')

new_density -= min(new_density)
new_density *= max(data_df["pdf"])/max(new_density)
# save smoothed output file
pickle.dump(np.array(new_density), open(f"3mnn_Y_smoothed_{NTag}b.p", 'wb'))

# get smoothed output for here
new_density = kde([fmp["m_h1"], fmp["m_h2"]])
new_density -= min(new_density)
new_density *= max(fmp["pdf"])/max(new_density)


fig = plt.figure()
ax = fig.add_subplot(111)
xmesh = np.array(fmp["m_h1"]).reshape((len(c.xbins),len(c.ybins))).T
ymesh = np.array(fmp["m_h2"]).reshape((len(c.xbins),len(c.ybins))).T
hmesh = np.array(new_density).reshape((len(c.xbins),len(c.ybins))).T
ax.pcolormesh(xmesh,ymesh,hmesh)
plot_functions.plotXhh()
plt.xlabel("$m_{h1}$")
plt.ylabel("$m_{h2}$")
plt.savefig(f"kde_2d_fullmassplane_{NTag}b_final.png")

pickle.dump(xmesh, open(f"xmesh_kde_2d_{NTag}b.p", 'wb'))
pickle.dump(ymesh, open(f"ymesh_kde_2d_{NTag}b.p", 'wb'))
pickle.dump(hmesh, open(f"hmesh_kde_2d_{NTag}b.p", 'wb'))

plt.cla(); plt.clf()
plt.scatter(fmp["pdf"], new_density)
plt.plot(fmp["pdf"],fmp["pdf"], 'k')
plt.xlabel("original pdf")
plt.ylabel("pdf after KDE")
plt.savefig(f"kde_2d_scatter_{NTag}b.png")
