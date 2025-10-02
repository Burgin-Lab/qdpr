QDPR
==============================
[//]: # (Badges)
<!-- [![GitHub Actions Build Status](https://github.com/Burgin-Lab/qdpr/workflows/CI/badge.svg)](https://github.com/Burgin-Lab/qdpr/actions?query=workflow%3ACI) -->
<!--[![codecov](https://codecov.io/gh/Burgin-Lab/QDPR/branch/main/graph/badge.svg)](https://codecov.io/gh/Burgin-Lab/QDPR/branch/main) -->


Implementation of quantified dynamics-property predictions for protein engineering campaigns.

In its current form, this code is not polished for general use and is provided only for transparency and repeatability in association with the paper Quantified Dynamics-Property Relationships: Data-Efficient Protein Engineering with Machine Learning of Protein Dynamics. A future, polished, better documented and all around improved version is planned.

Additional supporting scripts and data referred to in that paper are available at: https://zenodo.org/records/16377496

### Usage

Add the location of the downloaded repo to your python path. Invoke `qdpr.py` as:

```
python /path/to/qdpr.py configuration.config
```

Where `configuration.config` is the configuration file for the job you want to run. Configuration files that reproduce the work in the aforementioned publication are provided in the `qdpr/data` folder. Documentation of the available options in the configuration file is available in `configuration.txt`.

### Copyright

Copyright (c) 2025, Tucker Emme Burgin


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
