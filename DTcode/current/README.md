# Current pKa Predictor Version

This folder contains the current source code and datasets used for the manuscript package.

## Files

| File | Description |
| --- | --- |
| `predictpka260324.py` | Python source code for the graphical pKa prediction teaching software. |
| `dataset1.csv` | Case 1 dataset: baseline feature representation. |
| `dataset2.csv` | Case 2 dataset: adds coarse halogen information. |
| `dataset3.csv` | Case 3 dataset: expands atom/group count descriptors. |
| `dataset4.csv` | Case 4 dataset: adds electron-withdrawing group position and flag descriptors. |
| `dataset5.csv` | Case 5 dataset: compact chemistry-aware descriptors including `EWG_Pos` and `EWG_Rank`. |

## Windows executable

The Windows executable is distributed through figshare rather than stored in this GitHub repository:

[https://doi.org/10.6084/m9.figshare.29755451](https://doi.org/10.6084/m9.figshare.29755451)

## Fixed external test molecules

The fixed external test set contains four molecules:

- Octanoic acid
- 3-Chlorobutanoic acid
- Iodoacetic acid
- Phenylglyoxylic acid

These molecules are used to assess external prediction and should not be included in model training or hyperparameter selection.

## Data source

Molecular structures and pKa-related records were collected from PubChem and curated by the authors into the teaching datasets in this folder.

## Version note

This folder supersedes earlier development files in `DTcode`. Earlier files are preserved only for historical traceability.
