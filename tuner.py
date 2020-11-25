import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

import pickle

# get data
x = pickle.load(open('x.p', 'rb'))
y = pickle.load(open('y.p', 'rb'))
x, val_x = x[:200000, :], x[200000:, :]
y, val_y = y[:200000], y[200000:]


# function to build model given hp = hyperparameters to be tuned
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units_first',
                                        min_value=8,
                                        max_value=128,
                                        step=8),
                           input_dim=3,
                           activation='relu'))
    for i in range(5):
        model.add(layers.Dense(units=hp.Int(f'units{i}',
                                            min_value=32,
                                            max_value=1024,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='exponential'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])),
        loss='poisson',
        metrics=['accuracy'])
    return model

# make tuner, print search space, then actually search
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=15,
    executions_per_trial=3,
    directory='tuner_output',
    project_name='tuner')
tuner.search_space_summary()
tuner.search(x, y,
             epochs=50,
             validation_data=(val_x, val_y)
            )

# get best model
model = tuner.get_best_models(num_models=1)[0]
model.save('tuned_model')
print("done running!")
