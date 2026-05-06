# pKa-ML Education

No-code decision-tree software and teaching datasets for an undergraduate laboratory experiment on organic acid pKa prediction.

## Current manuscript version

The current source-code and dataset version used for the manuscript package is stored in:

`DTcode/current`

Current files:

- `predictpka260324.py`
- `dataset1.csv` to `dataset5.csv`

The Windows executable is distributed separately through figshare:

[https://doi.org/10.6084/m9.figshare.29755451](https://doi.org/10.6084/m9.figshare.29755451)

Earlier files in `DTcode` are retained as legacy development materials. For manuscript reproduction, use only `DTcode/current` unless otherwise noted.

## What this project teaches

This project introduces first-year undergraduate chemistry students to machine learning through a familiar chemical problem: predicting organic acid pKa values. The experiment uses an interpretable decision-tree model and a graphical interface so that students can complete the workflow without programming.

Students practice:

- translating molecular structures into numerical descriptors;
- comparing training, validation, and fixed external test performance;
- understanding overfitting through decision-tree depth;
- interpreting decision-tree splits and feature importance in chemical language;
- improving model generalization through chemistry-guided feature engineering.

## Dataset progression

The five datasets correspond to progressively improved feature-engineering cases.

| File | Teaching role |
| --- | --- |
| `dataset1.csv` | Baseline descriptors using simple carbon and carboxyl counts. |
| `dataset2.csv` | Adds a coarse halogen-count descriptor. |
| `dataset3.csv` | Expands substituent descriptors into element/group-specific counts. |
| `dataset4.csv` | Adds electron-withdrawing group position and flag descriptors. |
| `dataset5.csv` | Uses compact chemistry-aware descriptors, including `EWG_Pos` and `EWG_Rank`. |

Molecular structures and pKa-related records were collected from PubChem and curated by the authors for teaching use. The curated teaching data files are provided in `DTcode/current`.

The fixed external test set contains four molecules:

- Octanoic acid
- 3-Chlorobutanoic acid
- Iodoacetic acid
- Phenylglyoxylic acid

These molecules should remain outside model training and hyperparameter selection.

## How to run

### Option 1: Windows executable

Download the executable from figshare:

[https://doi.org/10.6084/m9.figshare.29755451](https://doi.org/10.6084/m9.figshare.29755451)

Then run the downloaded executable on Windows.

### Option 2: Python source

Clone this repository and run:

```bash
python DTcode/current/predictpka260324.py
```

Typical dependencies include:

- Python
- tkinter
- pandas
- numpy
- scikit-learn
- matplotlib
- Pillow
- RDKit

RDKit installation depends on the local Python environment. If the executable is available, Windows users can use it directly without manually installing these packages.

## Recommended citation

If you use this software or dataset in teaching or research, please cite the associated manuscript when available:

Yuming Su, Siman Cheng, Cheng Wang, and Yanping Ren. Interpretable machine learning for organic acid pKa prediction: a no-code feature-engineering laboratory experiment for first-year undergraduates. Manuscript in preparation.

## License

See `LICENSE`.
