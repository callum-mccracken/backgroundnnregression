import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import constants as c
import pandas as pd
from matplotlib import patches
from tqdm import tqdm
import os

def plotSR(m_h1_0=c.m_h1_0, m_h2_0=c.m_h2_0, r=c.r, Xhh_cut=c.Xhh_cut,
            m_h1_min=c.m_h1_min, m_h1_max=c.m_h1_max, m_h2_min=c.m_h2_min,
            m_h2_max=c.m_h2_max, color=c.sr_color):
    """
    Plots a given Xhh curve within m_h1_min,m_h1_max,m_h2_min,m_h2_max
    default parameter values are from the big paper
    """
    m_h1, m_h2 = sp.symbols('m_h1 m_h2')
    sg_expr = ((m_h1-m_h1_0)/(r*m_h1))**2 + ((m_h2-m_h2_0)/(r*m_h2))**2
    sg_eq = sp.Eq(sg_expr, Xhh_cut**2)
    plot = sp.plot_implicit(sg_eq,
                            x_var = (m_h1, m_h1_min, m_h1_max),
                            y_var = (m_h2, m_h2_min, m_h2_max),
                            show = False,
                            axis_center = (m_h1_min,m_h2_min))
    x,y = zip(*[(x_int.mid, y_int.mid)
        for x_int, y_int in plot[0].get_points()[0]])
    x,y = list(x),list(y)
    plt.plot(x,y,'.',markersize=0.5,color=color)

def plotVR(m_h1_0=c.m_h1_0, m_h2_0=c.m_h2_0,
           r=30, color=c.vr_color, n=500):
    theta = np.linspace(0, 2*np.pi, n)
    x1 = m_h1_0 + r*np.cos(theta)
    x2 = m_h2_0 + r*np.sin(theta)
    plt.plot(x1, x2, '.', markersize=0.5, color=color)

def plotCR(m_h1_0=c.m_h1_0, m_h2_0=c.m_h2_0,
           r=45, color=c.cr_color, n=500):
    plotVR(m_h1_0=m_h1_0, m_h2_0=m_h2_0,
           r=r, color=color, n=n)

def plot_fullmassplane_from_df(df, savename='fullmassplane.png',
                       save=True, show=False, vr=False):
    """plot the massplane for a given dataframe"""

    # add up all mhh for each bin, store in a dataframe
    assert all([x in df.keys() for x in ['pdf', 'm_h1', 'm_h2']])
    row_list = []
    for xi in tqdm(c.xbins):
        for yi in c.ybins:
            row_list.append({
                "m_h1": xi,
                "m_h2": yi,
                "pdf": sum(df.loc[
                    (df["m_h1"]==xi) & (df["m_h2"]==yi), "pdf"])
                })
    plot_df = pd.DataFrame(row_list)

    # cast m_h1, m_h2, pdf as arrays
    shape = (len(c.xbins),len(c.ybins))
    xmesh = np.array(plot_df["m_h1"]).reshape(shape).transpose()
    ymesh = np.array(plot_df["m_h2"]).reshape(shape).transpose()
    hmesh = np.array(plot_df["pdf"]).reshape(shape).transpose()

    # basic plot set-up
    fig, ax = plt.subplots()
    plt.xlabel("$m_{h1}$")
    plt.ylabel("$m_{h2}$")

    # plot SR outline
    plotSR()
    if vr:
        plotVR()

    # use hmesh as colors
    im = ax.pcolormesh(xmesh, ymesh, hmesh)
    fig.colorbar(im, ax=ax)

    # save figure if needed
    if save:
        plt.savefig(savename)
    if show:
        plt.show()
    return xmesh, ymesh, hmesh


def plot_regions():
    plotCR(n=2000)
    plotVR(n=2000)
    plotSR()
    plt.xlim(50, 250)
    plt.ylim(40, 200)

    plt.text(110, 110, "SR", color=c.sr_color, fontsize=20)
    plt.text(110,  82, "VR", color=c.vr_color, fontsize=20)
    plt.text(110,  68, "CR", color=c.cr_color, fontsize=20)

    plt.title("Control, Validation, and Signal Regions")

    plt.xlabel("$m_{H_1}$", fontsize=12)
    plt.ylabel("$m_{H_2}$", fontsize=12)
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig("figures/regions.png")


if __name__ == "__main__":
    plot_regions()

