import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import plot_functions

training=False
ModelName = "model_20_288_384_416_512_192_30e_25x25_poisson"

# Now lets make the regression model and train
def build_model():
    model = Sequential()
    model.add(Dense(20, input_dim=3, activation='relu'))
    model.add(Dense(288, activation='relu'))
    model.add(Dense(384, activation='relu'))
    model.add(Dense(416, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(192, activation='relu'))
    model.add(Dense(1, activation='exponential')) # Poisson loss requires output > 0 always
    model.compile(loss='poisson', optimizer='adam')
    return model

def create(data_df=None, X=None, Y=None):
    # Shuffle data points, so no training biasing/validation biasing
    np.random.seed(1234)
    if data_df is not None:
        data_df = data_df.sample(frac=1).reset_index(drop=True)
        X = data_df[["mh1","mh2","mhh"]]
        Y = data_df["pdf"]
    else:
        assert X is not None and Y is not None

    # Scale data to make NN easier
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    pickle.dump(scaler,open("MinMaxScaler4b.p",'wb'))

    if training:
        model = build_model()
        print(model.summary())
        history = model.fit(
        X_scaled, Y,
        epochs=30, validation_split = 0.1, verbose=2, shuffle=True)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.legend(["Training Set","Validation"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.savefig(ModelName+"_Loss.pdf")
        plt.show()

        model.save(ModelName)
    else:
        model = keras.models.load_model(ModelName)
        print(model.summary())

    model_output = model.predict(X_scaled).reshape(len(X_scaled))
    print(model_output.shape)
    print(X.shape)
    model_df = pd.DataFrame()
    model_df['pdf'] = model_output
    model_df['m_h1'] = X[:,0]
    model_df['m_h2'] = X[:,1]
    plot_functions.plot_fullmassplane_from_df(model_df, "nn_massplane_not_smoothed.png", show=True)

if __name__ == "__main__":
    X = np.array(np.array(pd.read_pickle("3mnn_X.p")))
    #print(X.shape)
    #Y = np.array(np.array(pd.read_pickle("3mnn_Y_smoothed.p")))
    Y = np.array(np.array(pd.read_pickle("3mnn_Y.p")))
    
    create(None, X, Y)