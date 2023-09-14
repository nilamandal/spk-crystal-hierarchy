import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

digits = load_digits()

digits_df = pd.DataFrame(digits.data[:,1:11])
digits_df['digit'] = pd.Series(digits.target).map(lambda x: 'Digit {}'.format(x))
#sns.pairplot(digits_df, hue='digit', palette='Spectral');
reducer = umap.UMAP(random_state=42)
reducer.fit(digits.data)


embedding = reducer.transform(digits.data)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))
print(embedding.shape)


plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24);

plt.show()
# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
# penguins = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv")
# penguins = penguins.dropna()
#
# #sns.pairplot(penguins.drop("year", axis=1), hue='species')
# #plt.show()
#
# reducer = umap.UMAP()
#
# penguin_data = penguins[
#     [
#         "bill_length_mm",
#         "bill_depth_mm",
#         "flipper_length_mm",
#         "body_mass_g",
#     ]
# ].values
# scaled_penguin_data = StandardScaler().fit_transform(penguin_data)
# embedding = reducer.fit_transform(scaled_penguin_data)
#
# print(embedding.shape)
#
# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=[sns.color_palette()[x] for x in penguins.species.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the Penguin dataset', fontsize=24);
#
# plt.show()
