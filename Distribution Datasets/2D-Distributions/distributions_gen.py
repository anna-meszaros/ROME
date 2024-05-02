#%%
import os
import numpy as np
import pickle

from sklearn import datasets

#%% Generate train_set
n_samples = 20000
random_state = 100

## Multi-modal distributions
# Dataset with two moons
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)[0]

# Anisotropicly distributed data
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation_withRot = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation_withRot)
aniso = X_aniso

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)[0]


## Uni-modal distributions
# Standard Normal
standard_normal = np.random.multivariate_normal([0,0],[[1,0],[0,1]], n_samples)

# Elliptical
transformation_noRot = [[1, 0], [0, 0.3]]
X_elliptical = np.dot(standard_normal, transformation_noRot)
elliptical = X_elliptical

# Rotated Elliptical
X_rotated_elliptical = np.dot(standard_normal, transformation_withRot)
rotated_elliptical = X_rotated_elliptical


#%%
os.makedirs('./Distribution Datasets/2D-Distributions/Processed_Data', exist_ok = True)
pickle.dump(noisy_moons, open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_moons_' + str(n_samples) + 'samples', 'wb'))
pickle.dump(aniso, open('./Distribution Datasets/2D-Distributions/Processed_Data/aniso_' + str(n_samples) + 'samples', 'wb'))
pickle.dump(varied, open('./Distribution Datasets/2D-Distributions/Processed_Data/varied_' + str(n_samples) + 'samples', 'wb'))

pickle.dump(standard_normal, open('./Distribution Datasets/2D-Distributions/Processed_Data/standard_normal_' + str(n_samples) + 'samples', 'wb'))
pickle.dump(elliptical, open('./Distribution Datasets/2D-Distributions/Processed_Data/elliptical_' + str(n_samples) + 'samples', 'wb'))
pickle.dump(rotated_elliptical, open('./Distribution Datasets/2D-Distributions/Processed_Data/rotated_elliptical_' + str(n_samples) + 'samples', 'wb'))
