[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ROME
This package implements the [Robust Multi-Modal Density Estimation](https://arxiv.org/abs/2401.10566). 
This method creates a probability density function using nonparametric methods.
ROME clusters data based on the OPTICS algortihm, using it to order sample points based on rechability analysis.
It then generates the clustering which maximizes the silhouette score.
Each cluster is then decorrelated and normalized before applying kernel density estimation.

ROME has shown itself to be superior when dealing with multi-modal and highly correlated distributions, especially for
high dimensions.

## Installation
ROME can be installed using:
```
pip install romepy
```

## Usage
Inside a script, ROME is loaded by the following command:
```
from rome.ROME import ROME

rome = ROME()
```

After initiliazing ROME, one can then call the typical functions most scikit-learn density estimators (KDE, GMM, etc.) have:
```
rome.fit(X, cluster=None) # fit distribution to data X
log_probs = rome.score_samples(X) # obtain log_probs of data X according to fitted distribution
X_new = rome.sample(num_samples = 10, random_state = 0) # generate new samples according to fitted distribution
```

The main difference is the *cluster* parameter in the *fit* function. If it is not set to *None*, it can be used to skip the OPTICS based clustering and use a predifened clustering instead.


## Changelog
Version 0.1.2: Initial upload of romepy.
