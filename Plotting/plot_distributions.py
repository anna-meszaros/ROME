#%% Import libraries
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sys.path.append('../')
from ROME.Prob_function import OPTICS_GMM



#%% Load the data

# 2D-Distributions
np.random.seed(0)
# Noisy Moons
noisy_moons = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_moons_20000samples', 'rb'))

# Varied
varied = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/varied_20000samples', 'rb'))

# Anisotropic
aniso = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/aniso_20000samples', 'rb'))

Datasets = {'Aniso': aniso, 'Varied': varied, 'Two Moons': noisy_moons}

# #%% Plot the distributions
n = 3000    
# # 2D-Distributions
# # Figure with 5 subplots in one line without axes

# # fig, axs = plt.subplots(1, len(Datasets), figsize=(len(Datasets) * 3, 3))

# for i, name in enumerate(Datasets):
#     print('Extracting ' + name)
#     # Get data
#     data = Datasets[name]
#     np.random.shuffle(data)
#     data = data[:n]

#     # Get clusters
#     print('Clustering ' + name)
#     Optics = OPTICS_GMM().fit(data)
#     cluster = Optics.cluster_labels 

#     # Get colors
#     colors = sns.color_palette("husl", cluster.max() + 1)
#     colors.append((0.0, 0.0, 0.0))
#     data_colors = [colors[i] for i in cluster]

#     # Plot
#     print('Plotting ' + name)
#     fig = plt.figure(i, figsize=(3, 3))
#     # plt.scatter(data[:, 0], data[:, 1], s=1, c=data_colors, alpha=0.9)
#     plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.1)
#     # plt.set_title(name)
#     plt.axis('equal')
#     plt.xticks([])
#     plt.yticks([])
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.gca().spines['left'].set_visible(False)
#     plt.gca().spines['bottom'].set_visible(False)

#     plt.show()

#     # Save figure as pdf
#     fig.savefig('./Distribution Datasets/2D-Distributions/Plots/' + name + '.pdf', bbox_inches='tight')



# # %%
# # Trajectories

# # Multivariate Bi-Modal Distribution obtained by augmenting recorded pedestrian trajectories
# print('Extracting Trajectories')
# Trajectories = pickle.load(open('./Distribution Datasets/Forking_Paths/Processed_Data/trajectories_20000samples', 'rb'))
# # shuffle the trajectories
# np.random.shuffle(Trajectories)
# Trajectories = Trajectories[:n]
# # Figure with 1 subplot

# print('Clustering Trajectories')
# # Optics = OPTICS_GMM().fit(Trajectories.copy().reshape(len(Trajectories), -1))
# # cluster = Optics.cluster_labels 

# # # Get colors
# # colors = sns.color_palette("husl", cluster.max() + 1)
# # colors.append((0.0, 0.0, 0.0))
# # data_colors = [colors[i] for i in cluster]

# fig = plt.figure() #figsize=(5, 3))
# for i in range(n):
#     # plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], c=data_colors[i], alpha=0.05)
#     plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], alpha=0.05, c='#1f77b4')
# # plt.title('Multi-Modal Trajectories')

# # set axis equal
# plt.axis('equal')

# # provide labels
# # plt.xlabel('$x\; [m]$')
# # plt.ylabel('$y\; [m]$')
# plt.xlim(3, 8)
# plt.ylim(4, 8.5)
# plt.gca().set_adjustable("box")
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.xticks([])
# plt.yticks([])

# plt.show()

# # Remove all spines
# fig.savefig('./Distribution Datasets/Forking_Paths/Plots/Trajectories.pdf', bbox_inches='tight')

#%%

# # Multivariate Bi-Modal Distribution obtained by augmenting recorded pedestrian trajectories
# print('Extracting Trajectories Uni-Modal')
# Trajectories = pickle.load(open('./Distribution Datasets/Forking_Paths/Processed_Data/trajectoriesUniModal_20000samples', 'rb'))
# # shuffle the trajectories
# np.random.shuffle(Trajectories)
# Trajectories = Trajectories[:n]
# # Figure with 1 subplot

# print('Clustering Trajectories Uni-Modal')

# fig = plt.figure()
# for i in range(n):
#     plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], alpha=0.05, c='#1f77b4')

# # set axis equal
# plt.axis('equal')

# # provide labels
# plt.xlim(3, 8)
# plt.ylim(4, 8.5)
# plt.gca().set_adjustable("box")
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.xticks([])
# plt.yticks([])

# plt.show()

# # Remove all spines
# fig.savefig('./Distribution Datasets/Forking_Paths/Plots/TrajectoriesUniModal.pdf', bbox_inches='tight')


# Gaussian
gaussianUniMod = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/gaussianUniModal_20000samples', 'rb'))

# Anisotropic
anisoUniMod = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/anisoUniModal_20000samples', 'rb'))

# Anisotropic rotated
anisoRotatedUniMod = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/anisoRotatedUniModal_20000samples', 'rb'))

# Datasets = {'AnisoUniModal': anisoUniMod, 'GaussianUniModal': gaussianUniMod, 'AnisoRotatedUniModal': anisoRotatedUniMod}
Datasets = {'AnisoRotatedUniModal': anisoRotatedUniMod}

#%% Plot the distributions
n = 3000    
# 2D-Distributions
# Figure with 5 subplots in one line without axes


for i, name in enumerate(Datasets):
    print('Extracting ' + name)
    # Get data
    data = Datasets[name]
    np.random.shuffle(data)
    data = data[:n]

    # Plot
    print('Plotting ' + name)
    fig = plt.figure(i, figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.1)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.show()

    # Save figure as pdf
    fig.savefig('./Distribution Datasets/2D-Distributions/Plots/' + name + '.pdf', bbox_inches='tight')
