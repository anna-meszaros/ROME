#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import datasets

#%% Generate train_set
n_samples = 20000
random_state = 100

# Dataset with two moons
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)[0]

# Anisotropicly distributed data
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = X_aniso

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)[0]


#%%
os.makedirs('./Distribution Datasets/2D-Distributions/Processed_Data', exist_ok = True)
pickle.dump(noisy_moons, open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_moons_' + str(n_samples) + 'samples', 'wb'))
pickle.dump(aniso, open('./Distribution Datasets/2D-Distributions/Processed_Data/aniso_' + str(n_samples) + 'samples', 'wb'))
pickle.dump(varied, open('./Distribution Datasets/2D-Distributions/Processed_Data/varied_' + str(n_samples) + 'samples', 'wb'))
