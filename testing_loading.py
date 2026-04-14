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


reconstructed_model = keras.models.load_model("debug/my_model.keras")
tr_input, tr_target= load_tr.__next__()
dummy_input, a, i= tr_input

pred= reconstructed_model.predict(dummy_input)

print(np.mean((dummy_input-pred)**2))
