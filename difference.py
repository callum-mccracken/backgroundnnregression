import matplotlib.pyplot as plt 
import pandas as pd
import plot_functions
import numpy as np

# get difference between 2b and 4b nn performance in VR

X = np.array(pd.read_pickle("3mnn_X.p"))

output_4b = np.array(pd.read_pickle("model_output_4b.p"))
output_2b = np.array(pd.read_pickle("model_output_2b.p"))
output_2b *= np.sum(output_4b)/np.sum(output_2b)



model_df = pd.DataFrame()
model_df['pdf'] = output_2b - output_4b
model_df['m_h1'] = X[:,0]
model_df['m_h2'] = X[:,1]
plot_functions.plot_fullmassplane_from_df(model_df, "massplane_2b_minus_4b.png", show=True, vr=True)

model_df = pd.DataFrame()
model_df['pdf'] = output_2b
model_df['m_h1'] = X[:,0]
model_df['m_h2'] = X[:,1]
plot_functions.plot_fullmassplane_from_df(model_df, "massplane_only_2b.png", vr=True)

model_df = pd.DataFrame()
model_df['pdf'] = output_4b
model_df['m_h1'] = X[:,0]
model_df['m_h2'] = X[:,1]
plot_functions.plot_fullmassplane_from_df(model_df, "massplane_only_4b.png", vr=True)