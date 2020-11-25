import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random
import pickle
from matplotlib.colors import ListedColormap

# load values and pdf
values = np.array(np.array(pd.read_pickle("X.p")).T)
density = np.array(pd.read_pickle("Y.p"))

plot = True

if plot:
    # take k random samples to make runtime manageable
    k = 10000
    indices = random.sample(range(len(density)), k)
    values = values[:,indices]
    density = density[indices]
    density /= max(density)

    
    # Make custom colormap so alpha value is 0 at min of range
    cmap = plt.cm.get_cmap("Greys")
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    alpha_cmap = ListedColormap(alpha_cmap)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 16), subplot_kw=dict(projection='3d'))
    im = axes[0].scatter(*values, c=[d**2 for d in density], cmap=alpha_cmap, marker='.')
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Before KDE")
    axes[0].set_xlim(min(values[0]),max(values[0]))
    axes[0].set_ylim(min(values[1]),max(values[1]))
    axes[0].set_zlim(min(values[2]),max(values[2]))
    axes[0].set_xlabel("$m_{h1}$")
    axes[0].set_ylabel("$m_{h2}$")
    axes[0].set_zlabel("$m_{hh}$")


    # do kde to get new density
    kde = stats.gaussian_kde(values)
    new_density = kde(values)
    # scale new_density 
    new_density -= min(new_density)
    new_density *= max(density)/max(new_density)

    print(max(new_density), max(density))

    # new scatterplot
    im = axes[1].scatter(*values, c=[d**2 for d in new_density], cmap=alpha_cmap, marker='.')
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title(f"After KDE, Fit To {k} Points")
    axes[1].set_xlim(min(values[0]),max(values[0]))
    axes[1].set_ylim(min(values[1]),max(values[1]))
    axes[1].set_zlim(min(values[2]),max(values[2]))
    axes[1].set_xlabel("$m_{h1}$")
    axes[1].set_ylabel("$m_{h2}$")
    axes[1].set_zlabel("$m_{hh}$")

    # show both
    plt.tight_layout()
    plt.savefig(f"kde_{k}.png")
    print('finished plotting')
    plt.show()

    plt.scatter(density, new_density)
    plt.xlabel("original pdf")
    plt.ylabel("pdf after KDE")
    plt.savefig(f"kde_scatter_{k}.png")
    plt.savefig(f"kde_scatter_{k}.png")
    plt.show()


    # re-open the files to start again
    values = np.array(np.array(pd.read_pickle("X.p")).T)
    density = np.array(pd.read_pickle("Y.p"))
else:
    if False:
        print('skipping plot, heading straight for kde')
        kde = stats.gaussian_kde(values)

if False:
    # this takes about a minute if plot=True, longer otherwise
    new_density = kde(values)
    new_density -= min(new_density)
    new_density *= max(density)/max(new_density)
    print(max(new_density))

#else:
#    print("loading new_density from file")
#    new_density = pickle.load(open("new_Y.p", 'rb'))




#pickle.dump(new_density, open("new_Y.p", 'wb'))

