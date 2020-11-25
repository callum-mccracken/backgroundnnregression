import uproot
import pickle
import numpy as np
import pandas as pd

six_masses = False
pairagraph = True


if pairagraph:
    data_filename = "pairAGraph_SM_2b_all_17_NN_100_bootstraps_IQR.root"
else:
    data_filename = "data17_NN_100_bootstraps_IQR.root"



if six_masses:
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

    #df_master = uproot.open(data_filename)["fullmassplane"]
    #for k in sorted(df_master.keys()):
    #    print(k)
    #exit()

    # Subset of 2b events
    # Comparable stats to 4b (within factor of 2)
    df_master = uproot.open(data_filename)["fullmassplane"].pandas.df(read_list,entrystop=500000)
    df = df_master.loc[df_master["ntag"] == 2]
    df[["m_h1","m_h2","m_hh"]].to_pickle("data_2tag.p")

    # All 4b events
    df_master = uproot.open(data_filename)["fullmassplane"].pandas.df(read_list)
    df = df_master.loc[df_master["ntag"] == 4]
    df[["m_h1","m_h2","m_hh"]].to_pickle("data_4tag.p")

    # All 2b events, with reweighting
    read_list += ["NN_d24_weight_bstrap_med_17"]
    df_master = uproot.open(data_filename)["fullmassplane"].pandas.df(read_list)
    df = df_master.loc[df_master["ntag"] == 2]
    df[["m_h1","m_h2","m_hh","NN_d24_weight_bstrap_med_17"]].to_pickle("data_2tag_full.p")

    #######################################
    # NOW GET ALL THE VARIABLES
    #######################################

    def calculate_masses(d):
        # 1 = h1_j1
        # 2 = h1_j2
        # 3 = h2_j1
        # 4 = h2_j2
        pt_1 = np.array(d['pT_h1_j1'])
        pt_2 = np.array(d['pT_h1_j2'])
        pt_3 = np.array(d['pT_h2_j1'])
        pt_4 = np.array(d['pT_h2_j2'])
        eta_1 = np.array(d['eta_h1_j1'])
        eta_2 = np.array(d['eta_h1_j2'])
        eta_3 = np.array(d['eta_h2_j1'])
        eta_4 = np.array(d['eta_h2_j2'])
        phi_1 = np.array(d['phi_h1_j1'])
        phi_2 = np.array(d['phi_h1_j2'])
        phi_3 = np.array(d['phi_h2_j1'])
        phi_4 = np.array(d['phi_h2_j2'])
        # calculate invariant mass a la wikipedia https://en.wikipedia.org/wiki/Invariant_mass#Collider_experiments
        m_12 = 2 * pt_1 * pt_2 * (np.cosh(eta_1 - eta_2) - np.cos(phi_1 - phi_2))
        m_13 = 2 * pt_1 * pt_3 * (np.cosh(eta_1 - eta_3) - np.cos(phi_1 - phi_3))
        m_14 = 2 * pt_1 * pt_4 * (np.cosh(eta_1 - eta_4) - np.cos(phi_1 - phi_4))
        m_23 = 2 * pt_2 * pt_3 * (np.cosh(eta_2 - eta_3) - np.cos(phi_2 - phi_3))
        m_24 = 2 * pt_2 * pt_4 * (np.cosh(eta_2 - eta_4) - np.cos(phi_2 - phi_4))
        m_34 = 2 * pt_3 * pt_4 * (np.cosh(eta_3 - eta_4) - np.cos(phi_3 - phi_4))
        return m_12, m_13, m_14, m_23, m_24, m_34

    # Subset of 2b events
    # Comparable stats to 4b (within factor of 2)
    df_master = uproot.open(data_filename)["fullmassplane"].pandas.df(read_list,entrystop=500000)
    df = df_master.loc[df_master["ntag"] == 2]
    m_12, m_13, m_14, m_23, m_24, m_34 = calculate_masses(df)
    df['m_12'] = m_12
    df['m_13'] = m_13
    df['m_14'] = m_14
    df['m_23'] = m_23
    df['m_24'] = m_24
    df['m_34'] = m_34
    df[["m_12", "m_13", "m_14", "m_23", "m_24", "m_34", "m_h1", "m_h2", "m_hh"]].to_pickle("data_2tag_6masses.p")

    # All 4b events
    df_master = uproot.open(data_filename)["fullmassplane"].pandas.df(read_list)
    df = df_master.loc[df_master["ntag"] == 4]
    m_12, m_13, m_14, m_23, m_24, m_34 = calculate_masses(df)
    df['m_12'] = m_12
    df['m_13'] = m_13
    df['m_14'] = m_14
    df['m_23'] = m_23
    df['m_24'] = m_24
    df['m_34'] = m_34
    df[["m_12", "m_13", "m_14", "m_23", "m_24", "m_34", "m_h1", "m_h2", "m_hh"]].to_pickle("data_4tag_6masses.p")

else:
    read_list = [
        "m_h1",
        "m_h2",
        "m_hh",
        "ntag",
        ]
    # Subset of 2b events
    # Comparable stats to 4b (within factor of 2 
    if pairagraph:
        dfs = []
        for k in uproot.open(data_filename).keys():
            print(k)
            try:
                d = uproot.open(data_filename)[k].pandas.df(read_list,entrystop=500000)
                dfs.append(d)
            except:
                print('skipping')
            
        df_master = pd.concat(dfs)
        #df_master_ctl = uproot.open(data_filename)["control"].pandas.df(read_list,entrystop=500000)
        #df_master_val = uproot.open(data_filename)["validation"].pandas.df(read_list,entrystop=500000)
        #df_master_sig = uproot.open(data_filename)["sig"].pandas.df(read_list,entrystop=500000)
        #df_master = pd.concat([df_master_ctl, df_master_val, df_master_sig])
        df = df_master.loc[df_master["ntag"] == 2]
        df[["m_h1","m_h2","m_hh"]].to_pickle("PG_data_2tag.p")
    else:
        df_master = uproot.open(data_filename)["fullmassplane"].pandas.df(read_list,entrystop=500000)
    df = df_master.loc[df_master["ntag"] == 2]
    df[["m_h1","m_h2","m_hh"]].to_pickle("data_2tag.p")

    # All 4b events
    if pairagraph:
        dfs = []
        for k in uproot.open(data_filename).keys():
            d = uproot.open(data_filename)[k].pandas.df(read_list)
            dfs.append(d)
        df_master = pd.concat(dfs)
        
        #df_master_ctl = uproot.open(data_filename)["control"].pandas.df(read_list)
        #df_master_val = uproot.open(data_filename)["validation"].pandas.df(read_list)
        #df_master_sig = uproot.open(data_filename)["sig"].pandas.df(read_list)
        #df_master = pd.concat([df_master_ctl, df_master_val, df_master_sig])
        df = df_master.loc[df_master["ntag"] == 4]
        df[["m_h1","m_h2","m_hh"]].to_pickle("PG_data_4tag.p")
    else:
        df_master = uproot.open(data_filename)["fullmassplane"].pandas.df(read_list)
        df = df_master.loc[df_master["ntag"] == 4]
        df[["m_h1","m_h2","m_hh"]].to_pickle("data_4tag.p")

    # All 2b events, with reweighting
    read_list += ["NN_d24_weight_bstrap_med_17"]
    if pairagraph:
        dfs = []
        for k in uproot.open(data_filename).keys():
            d = uproot.open(data_filename)[k].pandas.df(read_list)
            dfs.append(d)
        df_master = pd.concat(dfs)
        
        #df_master_ctl = uproot.open(data_filename)["control"].pandas.df(read_list)
        #df_master_val = uproot.open(data_filename)["validation"].pandas.df(read_list)
        #df_master_sig = uproot.open(data_filename)["sig"].pandas.df(read_list)
        #df_master = pd.concat([df_master_ctl, df_master_val, df_master_sig])
        df = df_master.loc[df_master["ntag"] == 2]
        df[["m_h1","m_h2","m_hh","NN_d24_weight_bstrap_med_17"]].to_pickle("PG_data_2tag_full.p")
    else:
        df_master = uproot.open(data_filename)["fullmassplane"].pandas.df(read_list)
        df = df_master.loc[df_master["ntag"] == 2]
        df[["m_h1","m_h2","m_hh","NN_d24_weight_bstrap_med_17"]].to_pickle("data_2tag_full.p")
