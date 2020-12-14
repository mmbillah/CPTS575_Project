#import libraries
import os
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt

# load data
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

train_features.info()
train_features.head()

#list of feature names
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
len(GENES+CELLS)

#scale the feature values
for col in (GENES + CELLS):
    transformer = QuantileTransformer(random_state=0, output_distribution="normal")
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = train_features[col].values.reshape(vec_len, 1)
    transformer.fit(raw_vec)

    train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
    
data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

#fit the data to PCS
pca = PCA().fit(data)
#plot cumulative variance against number of components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


plt.rcParams["figure.figsize"] = (10,6)

fig, ax = plt.subplots()
xi = np.arange(1, 773, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='tab:pink', alpha =0.45)

plt.xlabel('Number of Components',fontsize=15, family='serif')
plt.xticks(np.arange(0, 750, step=100))
plt.ylabel('Cumulative variance (%)', fontsize=15, family='serif')
plt.title('The number of components needed to explain variance',fontsize=20, family='serif')

plt.axhline(y=0.80, color='tab:orange', linestyle='--')
plt.text(0.5, 0.75, '80%', color = 'black', fontsize=16, family='serif')
plt.axhline(y=0.95, color='tab:cyan', linestyle='--')
plt.text(0.5, 0.90, '95%', color = 'black', fontsize=16, family='serif')


ax.grid(axis='x',color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
#show ffigure
plt.show()
#save figure
fig.savefig('pca.pdf',format='pdf')
