# ROME

Code connected to the publication [Robust Multi-Modal Density Estimation](https://arxiv.org/abs/2401.10566).

In order to run the experiments underlying the publication, run `Prob_func_eval.py`.

## Data Extraction

In order to read the results in a tabular form, run `data_extraction_baseline.py` for the baselines Manifold Parzen Windows<sup>[1]</sup> and Vine Copulas<sup>[2]</sup>, and `data_extraction.py` for the ablation studies.

## Plotting

There are 5 plotting codes:

`plot_distributions.py`: provides a visualisation of the distributions used for fitting the density estimators.

`plot_sampled_distributions.py`: provides a visualisation of samples generated from the fitted density estimators overlayed on the original samples used for fitting.

`plot_transformed_distributions.py`: provides a visualisation of the different data transformation steps of ROME.

`plot_distribution_functions.py`: provides contour plots of the probability density functions.

`metric_visualisation.py`: provides box_plot visualisation of the obtained metric values.


## Publication Data

To obtain the data based on which the results were reported, please refer to the online dataset at [4TU.ResearchData](https://data.4tu.nl/datasets/61f283ae-c30c-42d1-9a7c-89b454e013b3).


## References

[1] Pascal Vincent and Yoshua Bengio. Manifold parzen windows. _Advances in Neural Information Processing Systems_, 15, 2002.

[2] Thomas Nagler and Claudia Czado. Evading the curse of dimensionality in nonparametric density estimation with simplified vine copulas. _Journal of Multivariate Analysis_, 151:69–89, 2016
