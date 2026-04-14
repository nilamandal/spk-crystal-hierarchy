import numpy as np
import keras
from keras import ops
from spektral.data import DisjointLoader
from spektral_essential_objects import AtomFeaDataset
import pandas as pd

keras.saving.get_custom_objects().clear()

train_csv= pd.read_csv('../Main_fol_Zintl/train_100_ternary.csv')
train_data= AtomFeaDataset(train_csv, '../Main_fol_Zintl/', 8, 12, 'r')
load_tr= DisjointLoader(train_data, batch_size=len(train_csv))

@keras.saving.register_keras_serializable()
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = keras.layers.Dense(1)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        return self.layer(x)

model= MyModel()
model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")



tr_input, tr_target= load_tr.__next__()
dummy_input, a, i= tr_input
print(dummy_input.shape)
dummy_target = np.random.random((3966, 1))

model.fit(dummy_input, dummy_target)
pred=model.predict(dummy_input)
print(np.mean((dummy_input-pred)**2))




# Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
model.save("debug/my_model.keras")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("debug/my_model.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(dummy_input), reconstructed_model.predict(dummy_input)
)
