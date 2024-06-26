#%% Import libraries
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from sklearn.decomposition import PCA

sys.path.append('../')
from ROME.Prob_function import ROME

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

#%% Plot the distributions
n = 3000    
min_std = 0.01
# 2D-Distributions
# Figure with 5 subplots in one line without axes

# fig, axs = plt.subplots(1, len(Datasets), figsize=(len(Datasets) * 3, 3))

for i, name in enumerate(Datasets):
    print('Extracting ' + name)
    # Get data
    data = Datasets[name]
    np.random.shuffle(data)
    data = data[:n]

    # Get clusters
    print('Clustering ' + name)
    Optics = ROME().fit(data)
    cluster = Optics.cluster_labels 

    # Get colors
    colors = sns.color_palette("husl", cluster.max() + 1)
    colors.append((0.0, 0.0, 0.0))
    data_colors = [colors[i] for i in cluster]

    # Plot
    print('Plotting ' + name)
    fig = plt.figure(i, figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 1], s=1, c=data_colors, alpha=0.9)
    # plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.1)
    # plt.set_title(name)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.show()

    # Save figure as pdf
    fig.savefig('./Distribution Datasets/2D-Distributions/Plots/' + name + '_clustered.svg', bbox_inches='tight')


    # Feature Decorrelation
    unique_labels, cluster_size = np.unique(cluster, return_counts = True)

    num_features = data.shape[1]
    # initialise rotation matrix for PCA
    means = np.zeros((len(unique_labels), num_features))
    T_mat = np.zeros((len(unique_labels), num_features, num_features))
        
    Stds = np.zeros((len(unique_labels), num_features)) 

    X_label_pca_all = np.zeros(data.shape)
    X_label_pca_norm_all = np.zeros(data.shape)

    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        # Get cluster data
        X_label = data[cluster == label]
        num_samples = len(X_label)
        assert num_samples == cluster_size[i]

        # Get mean and std
        means[i] = X_label.mean(0)
        Stds[i]  = X_label.std(0)

        # Shift coordinate system origin to mean
        X_label_stand = (X_label - means[[i]])
        
        # Repeat data if not enough samples are available
        if num_samples < num_features:
            c = np.tile(X_label_stand, (int(np.ceil(num_features / num_samples)),1))
        else:
            c = X_label_stand.copy()

        # calculate PCA on X_label_stand -> get rot matrix and std
        # Stabalize correlation matrix if necessary
        attempt = 0
        successful_pca = False
        while not successful_pca:
            try:
                pca = PCA(random_state = 0).fit(c)
                successful_pca = True
            except:
                e_fac = 10 ** (0.5 * attempt - 6)
                c[:num_features] += np.eye(num_features) * e_fac 
                        
                # Prepare next attempt
                attempt += 1
            
            if not successful_pca:
                print('PCA failed, was done again with different random start.')
            
        # Exctract components std
        pca_std = np.sqrt(pca.explained_variance_)

        # Extract roation matrix
        T_mat[i]  = pca.components_.T


        # Apply transformation matrix
        X_label_pca = X_label_stand @ T_mat[i] # @ is matrix multiplication
        X_label_pca_all[cluster == label] = X_label_pca


        pca_std = pca_std * (pca_std.max() - min_std) / pca_std.max() + min_std
        
        # Adjust T_mat accordingly
        T_mat[i]         /= pca_std[np.newaxis]
        
        # Apply transformation matrix
        X_label_pca_norm = X_label_stand @ T_mat[i] # @ is matrix multiplication
        X_label_pca_norm_all[cluster == label] = X_label_pca_norm


    # Plot
    print('Plotting ' + name)
    plt.clf()
    fig = plt.figure(i, figsize=(3*len(unique_labels), 3))
    axs = []
    for i, label in enumerate(unique_labels):
        axs.append(fig.add_subplot(1, len(unique_labels), i+1))
        axs[i].scatter(X_label_pca_all[cluster == label, 0], X_label_pca_all[cluster == label, 1],
                        s=1, c=np.array(data_colors)[cluster == label], alpha=0.9)
        axs[i].axis('equal')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)

    axs_id_largestLim = 0
    for i in range(len(axs)):
        if axs[i].get_xlim()[1] - axs[i].get_xlim()[0] > axs[axs_id_largestLim].get_xlim()[1] - axs[axs_id_largestLim].get_xlim()[0]:
            axs_id_largestLim = i
    
    for i, label in enumerate(unique_labels):
        plt.setp(axs[i], xlim=axs[axs_id_largestLim].get_xlim(), ylim=axs[axs_id_largestLim].get_ylim())

    plt.show()

    fig.savefig('./Distribution Datasets/2D-Distributions/Plots/' + name + '_decorrelated.svg', bbox_inches='tight')


    # Normalisation
    # Plot
    print('Plotting ' + name)
    plt.clf()
    fig = plt.figure(i, figsize=(3*len(unique_labels), 3))
    axs = []
    for i, label in enumerate(unique_labels):
        axs.append(fig.add_subplot(1, len(unique_labels), i+1))
        axs[i].scatter(X_label_pca_norm_all[cluster == label, 0], X_label_pca_norm_all[cluster == label, 1],
                        s=1, c=np.array(data_colors)[cluster == label], alpha=0.9)
        axs[i].axis('equal')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
    
    for i, label in enumerate(unique_labels):
        plt.setp(axs[i], xlim=axs[axs_id_largestLim].get_xlim(), ylim=axs[axs_id_largestLim].get_ylim())

    plt.show()

    fig.savefig('./Distribution Datasets/2D-Distributions/Plots/' + name + '_normalised.svg', bbox_inches='tight')




# %%
# Trajectories

# Multivariate Bi-Modal Distribution obtained by augmenting recorded pedestrian trajectories
print('Extracting Trajectories')
Trajectories = pickle.load(open('./Distribution Datasets/Forking_Paths/Processed_Data/trajectories_20000samples', 'rb'))
# shuffle the trajectories
np.random.shuffle(Trajectories)
Trajectories = Trajectories[:n]
# Figure with 1 subplot

print('Clustering Trajectories')
# Optics = ROME().fit(Trajectories.copy().reshape(len(Trajectories), -1))
# cluster = Optics.cluster_labels 

# # Get colors
# colors = sns.color_palette("husl", cluster.max() + 1)
# colors.append((0.0, 0.0, 0.0))
# data_colors = [colors[i] for i in cluster]

fig = plt.figure() #figsize=(5, 3))
for i in range(n):
    # plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], c=data_colors[i], alpha=0.05)
    plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], alpha=0.05, c='#1f77b4')
# plt.title('Multi-Modal Trajectories')

# set axis equal
plt.axis('equal')

# provide labels
# plt.xlabel('$x\; [m]$')
# plt.ylabel('$y\; [m]$')
plt.xlim(3, 8)
plt.ylim(4, 8.5)
plt.gca().set_adjustable("box")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])

plt.show()

# Remove all spines
fig.savefig('./Distribution Datasets/Forking_Paths/Plots/Trajectories.pdf', bbox_inches='tight')
