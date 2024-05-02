#%%
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import Prob_function as pf

from utils import create_random_data_splt


def plot_cluster_results(samples, labels, plot_file):
    colors = sns.color_palette("husl", labels.max() + 1)
    colors.append((0.0, 0.0, 0.0))
    
    fig = plt.figure()
    for label in np.unique(labels):
        sample_id = np.where(labels == label)[0]
        samples_label = samples[sample_id]
        plt.scatter(samples_label[:,0], samples_label[:,1], c = colors[label], s = 10)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    # Equal aspect ratio
    plt.axis('equal')
    plt.gca().set_axis_off()
    
    os.makedirs(os.path.dirname(plot_file), exist_ok = True)
    fig.savefig(plot_file + '.pdf', bbox_inches='tight')
    fig.savefig(plot_file + '.svg', bbox_inches='tight')
    fig.savefig(plot_file + '.png', bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_trajectory_cluster_results(samples, labels, plot_file):
    colors = sns.color_palette("husl", labels.max() + 1)
    colors.append((0.0, 0.0, 0.0))
    
    fig = plt.figure()
    for label in np.unique(labels):
        sample_id = np.where(labels == label)[0]
        samples_label = samples[sample_id].reshape(-1,12,2).transpose(1,0,2)
        plt.plot(samples_label[...,0], samples_label[...,1], c = colors[label], linewidth = 0.5)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    # Equal aspect ratio
    plt.axis('equal')
    plt.gca().set_axis_off()
    
    os.makedirs(os.path.dirname(plot_file), exist_ok = True)
    fig.savefig(plot_file + '.pdf', bbox_inches='tight')
    fig.savefig(plot_file + '.svg', bbox_inches='tight')
    fig.savefig(plot_file + '.png', bbox_inches='tight')
    plt.clf()
    plt.close()


def main():
    #%% Load the datasets
    # 2D-Distributions
    # Noisy Moons
    noisy_moons = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_moons_20000samples', 'rb'))

    # Varied
    varied = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/varied_20000samples', 'rb'))

    # Anisotropic
    aniso = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/aniso_20000samples', 'rb'))


    # Multivariate Bi-Modal Distribution obtained by augmenting recorded pedestrian trajectories
    Trajectories = pickle.load(open('./Distribution Datasets/Forking_Paths/Processed_Data/trajectories_20000samples', 'rb'))
    Trajectories = Trajectories.reshape(len(Trajectories), Trajectories.shape[1]*Trajectories.shape[2])

    #%% Create multiple datasets with different number of samples 
    # and save to dictionaries with keys containing info on dataset_name, n_samples and rand_seed

    num_samples = [200, 600, 2000, 6000]

    sample_str = './Hyperparameter_test/sampled_datasets_dict.pkl'

    sample_dict = {}

    print("", flush = True)
    print("Extracting datasets", flush = True)
    for n_samples in num_samples:
        key = 'n_samples_' + str(n_samples)

        sample_dict['Trajectories_' + key], _ = create_random_data_splt(Trajectories, 0, n_samples)
        sample_dict['noisy_moons_' + key], _ = create_random_data_splt(noisy_moons, 0, n_samples)
        sample_dict['varied_' + key], _ = create_random_data_splt(varied, 0, n_samples)
        sample_dict['aniso_' + key], _ = create_random_data_splt(aniso, 0, n_samples)
        
        os.makedirs(os.path.dirname(sample_str), exist_ok = True)
        pickle.dump(sample_dict, open(sample_str, 'wb'))


    #%% Define test cases
    # 2D-Distributions
    twoD_min_std = 0.01
    # Trajectory Distributions
    traj_min_std = 0.025

    # configs format: use_cluster, use_PCA, use_std, use_KDE, min_std
    use_PCA = True
    use_std = True

    testConfigs = [['silhouette', not use_PCA, use_std, 'KDE'],
                   [      'DBCV', not use_PCA, use_std, 'KDE']] 
    
    cluster_str = './Hyperparameter_test/cluster_dict.pkl'
    cluster_dict = {}
    
    print("", flush = True)
    print("Clustering", flush = True)
    
    for key, samples in sample_dict.items():
        for config in testConfigs:
            pf_key = key + '_' + config[0]
            
            if not('Trajectories' in key):
                min_std = twoD_min_std
            else:
                min_std = traj_min_std

            print('Fit distribution for ' + pf_key, flush = True) 

            distr_mdl = pf.ROME(use_cluster=config[0], use_PCA=config[1],
                                      use_std=config[2], estimator=config[3], 
                                      min_std=min_std)

            reach_file_plot = './Hyperparameter_test/Plots/Reachibility/' + pf_key
            distr_mdl.fit(samples, plot_reach_file=reach_file_plot)

            cluster_plot = './Hyperparameter_test/Plots/Clusters/' + pf_key + '_cluster_plot'

            labels = distr_mdl.cluster_labels

            if 'Trajectories' in key:
                plot_trajectory_cluster_results(samples, labels, cluster_plot)
            else:
                plot_cluster_results(samples, labels, cluster_plot)

            cluster_dict[pf_key] = distr_mdl
        
            pickle.dump(cluster_dict, open(cluster_str, 'wb'))


if __name__ == '__main__':
    main()
