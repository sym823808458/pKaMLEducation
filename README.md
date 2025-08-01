# Interactive pKa Prediction Tool for Chemistry Education

🎓 **A Zero-Programming Educational Tool for Teaching Machine Learning in Chemistry**

An interactive GUI application that teaches undergraduate chemistry students both acid-base chemistry and machine learning concepts through pKa prediction using decision trees. **No programming experience required** - just download and run!
✨ Key Features  
🎯 Zero Programming Barrier  
Complete graphical user interface (Tkinter-based)  
Built-in step-by-step tutorial covering ML concepts    
Click-and-explore interface for all functions  
No command line or coding required  
📚 Educational Components  
Interactive Knowledge Steps: 9-step tutorial covering acid-base chemistry and ML basics  
Dataset Visualization: Click molecules to see structures, explore pKa distributions  
Decision Tree Depth Selection: Visual explanation of model complexity  
Real-time Performance Metrics: MAE, MSE, R² with training vs validation comparison  
Overfitting Demonstration: See how model complexity affects generalization  
🧪 Chemical Integration  
SMILES to Structure: Automatic molecular visualization using RDKit  
Feature Engineering: Learn how molecular properties become ML features  
Chemical Interpretation: Connect decision tree splits to chemical logic   
🎓 Educational Objectives  
Students will learn to:  
✅ Understand decision tree machine learning models  
✅ Recognize and prevent overfitting through depth comparison  
✅ Evaluate model performance using standard metrics  
✅ Connect molecular structure to chemical properties  
✅ Interpret SMILES notation and molecular descriptors  
✅ Apply computational thinking to chemistry problems  
📊 Built-in Datasets  
Dataset 1: Aliphatic Carboxylic Acids  
18 molecules: Formic acid to trichloroacetic acid  
5 features: Carbon count, functional groups, substituents  
pKa range: 0.51 - 4.90  
Purpose: Introduction to structure-activity relationships  
Dataset 2: Aromatic Acids and Phenols  
18 molecules: Aromatic compounds with diverse substituents  
9 features: Including aromaticity, electronic effects, positions  
pKa range: 2.21 - 9.99  
Purpose: Advanced electronic effects and conjugation  


## 🚀 Quick Start (Choose Your Method)

### Option 1: Windows Users - Direct Download (Easiest!)
1. **Download the executable**: Go to [10.6084/m9.figshare.29755451](https://figshare.com/articles/software/predictpka250523_exe/29755451?file=56780045) and download `pKa_Predictor.exe`
2. **Double-click to run** - No installation needed!
3. **Start learning** - Follow the built-in tutorial

### Option 2: Python Users - Source Code
git clone https://github.com/sym823808458/pKaMLEducation.git
📁 Technical Details
Dependencies
tkinter          # GUI framework (built-in with Python)
pandas          # Data manipulation
numpy           # Numerical computing
rdkit           # Molecular informatics
PIL (Pillow)    # Image processing
scikit-learn    # Machine learning
matplotlib      # Plotting

### Option 2: Python Users - Jupyter notebook Code
see PredictpKa_notebook.ipynb

📄 Citation
If you use this tool in your research or teaching, please cite:
@article{su2024interactive,
  title={An Interactive and Interpretable Decision Tree Tool for Teaching pKa Prediction in Undergraduate Chemistry},
  author={Su, Yuming and Cheng, Siman and Wang, Cheng and Ren, Yanping},
  journal={Journal of Chemical Education},
  year={2024},
  note={In preparation}
}
