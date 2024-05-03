[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ROME
This package implements the [Robust Multi-Modal Density Estimation](https://arxiv.org/abs/2401.10566). 
This method is using clustering based on the OPTICS algortihm, using is to order sample points based on rechability analysis, based on which the clustering which maximizes the silhuette is selected.
On each cluster, it then uses normalization and decorrelation, before fitting applying simple KDE to it.

ROME has shown itself to be superior when dealing with multi-modal and highly corrolated distributions.

## Installation
ROME can be installed using the simple "pip install romepy" command.

## Usage
Inside a script, ROME is loaded by the following command.
```
from ROME import ROME

rome = ROME()
```

After initiliazing ROME, one can then call the typical functions most scikit-learn density estimators (KDE, GMM, etc.) have:
```
rome = rome.fit(X, cluster=None)
log_probs = rome.score_samples(X)
X_new = rome.sample(num_samples = 10, random_state = 0)
```

The main difference is the *cluster* parameter in the fit function. If it is not set to *None*, it can be used to skip the OPTICS based clustering and use a predifened clustering instead.


## Changelog
Version 0.1.0: Initial upload of romepy.
