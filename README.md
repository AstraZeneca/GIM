# GIM
![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

# Introduction
We developed Gene Interaction Matrices (GIM) as a biologically inspired, gene-interaction based data transformation on gene expression data to create an image-like feature matrix from any gene expression-based study. The transformed data can then be used with any CNN based machine learning approaches for a variety of challenging problems such as disease diagnostics and drug development.

# Dependencies
```
python >= 3.7
numpy 1.20.3
pandas 1.3.0
```

# Usage
An example workflow to create a GIM is described here. To begin import the necessary dependencies and gim .py file containing the transform function.
```
import pandas
import gim
```
Load the treatment and control files containing one or more replicates from the gim/data/ directory.
```
df_control_replicates = pd.read_csv("control_replicates.csv")
df_treatment_replicates = pd.read_csv("treatment_replicates.csv")
```
Apply the gim_transform function to the files.
```
sample_img = gim.gim_transform(df_control_replicates, df_treatement_replicates)
```

# Reference
TBD
