# How to Execute the Customer Purchase Prediction Project

Complete step-by-step instructions for setting up, running, and deploying the Customer Purchase Prediction project.

---

## Step 1: Create Project Folder

Create a new directory for the project:

```bash
mkdir Customer-Purchase-Prediction
cd Customer-Purchase-Prediction
```

Verify the directory structure:
```bash
ls -la
```

---

## Step 2: Initialize Git Repository (Optional but Recommended)

Initialize a Git repository for version control:

```bash
git init
git add .
git commit -m "Initial commit: Customer Purchase Prediction project"
```

Create a `.gitignore` file to exclude unnecessary files:

```bash
cat > .gitignore << EOF
venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
.DS_Store
*.egg-info/
dist/
build/
EOF
```

---

## Step 3: Add Dataset

Place your customer data CSV file in the `data/` directory:

```bash
# The notebook expects the file at: data/customer_data.csv
# File should have the following structure:
# - First row: column headers
# - Remaining rows: data
# - Last column: target variable (Purchase: 0 or 1)
```

### Dataset Requirements

- **Format**: CSV (comma-separated values)
- **Target Column**: Must be the last column
- **Target Values**: Binary (0 = No Purchase, 1 = Purchase)
- **Missing Values**: Acceptable (handled by notebook)
- **Minimum Features**: 2-3 numerical + categorical features

### Sample CSV Structure

```
Age,Income,TimeSpent,VisitCount,ClickThrough,Segment,DeviceType,Region,Purchase
25,45000,120,5,0.15,Standard,Mobile,North,0
34,62000,240,12,0.35,Premium,Desktop,South,1
28,51000,180,8,0.25,Standard,Mobile,East,1
```

### Preparing Your Own Dataset

If using your own data:

1. Ensure CSV format
2. Check for valid column names (no special characters)
3. Verify target variable is binary (0/1 or Yes/No)
4. Place file at: `data/customer_data.csv`
5. Run initial exploration in notebook cell 3

---

## Step 4: Create Virtual Environment

A virtual environment isolates project dependencies from system Python.

### On Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Verify Virtual Environment

```bash
which python  # Linux/macOS
where python  # Windows
```

You should see the path pointing to your venv directory.

---

## Step 5: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

Test that all packages import correctly:

```bash
python << EOF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("All imports successful!")
EOF
```

### Installing Individual Packages

If you prefer to install packages individually:

```bash
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.0
pip install jupyter==1.0.0
```

---

## Step 6: Open and Run Jupyter Notebook

### Start Jupyter Server

```bash
jupyter notebook
```

This will:
1. Start a local Jupyter server
2. Open your default web browser
3. Display the Jupyter file browser at `http://localhost:8888`

### Open the Notebook

1. Navigate to `Customer_Purchase_Prediction.ipynb` in the Jupyter interface
2. Click to open the notebook
3. The notebook will load in a new tab

### Jupyter Navigation Tips

- **Run cell**: Press `Ctrl+Enter` or click the play button
- **Run cell and move to next**: Press `Shift+Enter`
- **Add new cell**: Click "+" in toolbar or press `Esc` then `A` (above) or `B` (below)
- **Delete cell**: Select cell and press `Esc` then `D` twice
- **Markdown mode**: Press `Esc` then `M`
- **Code mode**: Press `Esc` then `Y`

---

## Step 7: Run Notebook Cells in Order

### Execution Sequence

The notebook must be executed sequentially (top to bottom) for proper operation:

#### **Section 1: Library Imports** (Run Once)
```
Cell: Import necessary libraries
- Imports pandas, numpy, matplotlib, scikit-learn
- Sets matplotlib style and seaborn palette
```

#### **Section 2: Load Dataset** (Run Once)
```
Cells: Load and explore data
- Reads data/customer_data.csv
- Displays dataset shape, head, info, describe
- Shows missing values statistics
```

#### **Section 3: Exploratory Data Analysis** (Run Once)
```
Cells: Analyze data patterns
- Target distribution visualization
- Numerical features histograms and boxplots
- Correlation heatmap
- Categorical features analysis
```

#### **Section 4: Data Preprocessing** (Run Once)
```
Cells: Prepare data for modeling
- Handle missing values (imputation)
- Encode categorical variables
- Scale numerical features
- Split into train-test sets
```

#### **Section 5: Model Training** (Run Once)
```
Cells: Train four classification models
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
```

#### **Section 6: Model Evaluation** (Run Once)
```
Cells: Evaluate model performance
- Generate predictions for each model
- Calculate accuracy, precision, recall, F1
- Display confusion matrices
- Show classification reports
```

#### **Section 7: Model Comparison** (Run Once)
```
Cells: Compare models side-by-side
- Create comparison dataframe
- Generate comparison visualization
- Rank models by performance
```

#### **Section 8: Feature Importance** (Run Once)
```
Cells: Analyze feature influence
- Random Forest feature importance
- Logistic Regression coefficients
- Visualizations for both
```

#### **Section 9: Conclusion** (Read Only)
```
Markdown summary of findings and recommendations
```

### Handling Execution Errors

If a cell fails:

1. **Check error message**: Read the traceback carefully
2. **Common issues**:
   - Dataset not found: Verify `data/customer_data.csv` exists
   - Missing packages: Run `pip install -r requirements.txt` again
   - Invalid data format: Check CSV structure and encoding
3. **Solutions**:
   - Check file paths (use absolute paths if relative fails)
   - Restart kernel: Kernel menu → Restart & Run All
   - Check data types: Run `df.info()` to inspect

### Restarting the Notebook

If you want to clear all outputs and run again:

1. Click `Kernel` menu
2. Select `Restart & Clear Output`
3. Click `Restart`
4. Run cells from the top in order

### Using Keyboard Shortcuts

For faster execution:
- `Ctrl+Shift+P`: Open command palette (macOS: `Cmd+Shift+P`)
- `Ctrl+A`: Select all cells
- `Ctrl+Shift+Enter`: Run all cells

---

## Step 8: Saving the Trained Model (Optional)

To save models for later use:

```python
import pickle

# After training models, add these cells:

# Save a model
with open('model_logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# To load later:
with open('model_logistic_regression.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### Creating a Prediction Function

Add this code cell for making predictions on new data:

```python
def predict_customer_purchase(customer_data_dict):
    """
    Predict purchase probability for a customer.
    
    Args:
        customer_data_dict: Dictionary with feature names and values
        
    Returns:
        Prediction and probability
    """
    # Convert dict to DataFrame
    customer_df = pd.DataFrame([customer_data_dict])
    
    # Apply same preprocessing
    customer_scaled = scaler.transform(customer_df)
    
    # Predict with best model
    prediction = rf_model.predict(customer_scaled)[0]
    probability = rf_model.predict_proba(customer_scaled)[0]
    
    return {
        'prediction': 'Will Purchase' if prediction == 1 else 'Will Not Purchase',
        'probability_no_purchase': probability[0],
        'probability_purchase': probability[1]
    }

# Example usage
customer_example = {
    'Age': 30,
    'Income': 60000,
    'TimeSpent': 200,
    # ... add other features
}
result = predict_customer_purchase(customer_example)
print(result)
```

---

## Step 9: Export Notebook as PDF

### Method 1: Using Jupyter Interface

1. Open the notebook in Jupyter
2. Click `File` menu
3. Select `Export Notebook As` → `Export Notebook to PDF`
4. Choose location and save

### Method 2: Using Command Line

```bash
# First install nbconvert if not already installed
pip install nbconvert

# Convert notebook to PDF
jupyter nbconvert --to pdf Customer_Purchase_Prediction.ipynb

# PDF will be saved as: Customer_Purchase_Prediction.pdf
```

### Method 3: Using Command Line (with options)

```bash
# Export with table of contents
jupyter nbconvert --to pdf \
  --output-dir=outputs \
  Customer_Purchase_Prediction.ipynb

# Export with custom template
jupyter nbconvert --to pdf \
  --template=classic \
  Customer_Purchase_Prediction.ipynb
```

### PDF Export Troubleshooting

If PDF export fails:

```bash
# Check if required packages are installed
pip install reportlab pillow

# For Windows users, may need pandoc:
# Download from: https://pandoc.org/installing.html
# Or install via conda: conda install pandoc
```

### Saving Visualizations as Images

To save individual plots:

```python
# Add this in notebook cells where plots are generated

fig.savefig('images/plot_name.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Step 10: Publish to GitHub

### Initial GitHub Setup

1. Create repository on GitHub.com
2. Name it: `Customer-Purchase-Prediction`
3. Add description
4. Choose public/private
5. Do NOT initialize with README (we have one)

### Pushing Code to GitHub

```bash
# Add remote repository
git remote add origin https://github.com/yourusername/Customer-Purchase-Prediction.git

# Verify remote
git remote -v

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Complete ML pipeline for customer purchase prediction"

# Push to GitHub (first time - use -u to set upstream)
git branch -M main  # Ensure main branch
git push -u origin main

# Future pushes
git push origin main
```

### Creating a GitHub Profile README

Create `GitHub_Profile_README.md` to showcase the project:

```markdown
# Customer Purchase Prediction Using Classification Algorithms

A machine learning project demonstrating a complete data science workflow for predicting customer purchase behavior.

## Key Features
- Multiple classification algorithms (Logistic Regression, Decision Tree, Random Forest, KNN)
- Comprehensive exploratory data analysis
- Complete data preprocessing pipeline
- Model comparison and evaluation
- Feature importance analysis

## Technologies
- Python 3.7+
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## Quick Start
See [README.md](README.md) for installation and usage instructions.

## Results
- Implements 4 classification models
- Evaluates using accuracy, precision, recall, F1-score
- Provides feature importance insights
- Production-ready codebase

## Author
[Your Name]

## License
MIT License - see LICENSE file for details
```

### GitHub Repository Best Practices

1. **Add .gitignore** (already created in Step 2)
2. **Write descriptive commit messages**
3. **Keep repo clean** (no data files unless necessary)
4. **Update README** with results
5. **Add badges** (optional but professional):

```markdown
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
```

### Sharing the Repository

```bash
# Clone URL for others
https://github.com/yourusername/Customer-Purchase-Prediction.git

# Instructions for others to run:
1. git clone <url>
2. cd Customer-Purchase-Prediction
3. python -m venv venv
4. source venv/bin/activate  # or venv\Scripts\activate on Windows
5. pip install -r requirements.txt
6. jupyter notebook Customer_Purchase_Prediction.ipynb
```

---

## Step 11: Project Verification Checklist

After completing all steps, verify:

- [ ] Virtual environment activated (`venv` in path)
- [ ] All dependencies installed (`pip list`)
- [ ] Dataset in `data/customer_data.csv`
- [ ] Notebook runs without errors
- [ ] All visualizations generate successfully
- [ ] Model evaluation metrics display correctly
- [ ] PDF export completes successfully
- [ ] Git repository initialized and committed
- [ ] GitHub repository created and pushed
- [ ] README file complete and accurate
- [ ] LICENSE file present
- [ ] requirements.txt lists all packages

---

## Step 12: Troubleshooting Common Issues

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Solution**: Install scikit-learn
```bash
pip install scikit-learn==1.3.0
```

### Issue: "FileNotFoundError: data/customer_data.csv"

**Solution**: Verify file exists
```bash
ls data/customer_data.csv  # Linux/macOS
dir data\customer_data.csv  # Windows
```

### Issue: Notebook runs slowly

**Solution**: Close other applications or restart kernel
```bash
# In notebook: Kernel → Restart & Run All
```

### Issue: Plots not displaying

**Solution**: Add matplotlib backend directive
```python
%matplotlib inline
```

### Issue: Virtual environment not activating

**Solution**: Verify correct path
```bash
# Linux/macOS
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### Issue: PDF export fails

**Solution**: Install additional dependencies
```bash
pip install pandoc nbconvert
```

---

## Step 13: Project Extension Ideas

After running the basic pipeline, consider:

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(knn_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X, y, cv=5, scoring='f1_weighted')
print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.4f}")
```

### Advanced Visualizations
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.show()
```

### Model Deployment
```python
# Create Flask API for predictions
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})
```

---

## Step 14: Performance Optimization

### Reduce Notebook Load Time
```python
# Subsample large datasets for development
df_sample = df.sample(frac=0.1, random_state=42)

# Reduce number of trees in Random Forest
rf_model = RandomForestClassifier(n_estimators=50)

# Reduce visualization quality for faster rendering
plt.rcParams['figure.dpi'] = 100
```

### Parallel Processing
```python
# Use all CPU cores for Random Forest
rf_model = RandomForestClassifier(n_jobs=-1)

# Use all cores for scikit-learn operations
from sklearn import config_context
with config_context(n_jobs=-1):
    model.fit(X_train, y_train)
```

---

## Quick Reference Commands

```bash
# Setup
mkdir Customer-Purchase-Prediction && cd Customer-Purchase-Prediction
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# Running
jupyter notebook Customer_Purchase_Prediction.ipynb

# Export
jupyter nbconvert --to pdf Customer_Purchase_Prediction.ipynb

# Git operations
git init
git add .
git commit -m "message"
git push origin main

# Cleanup
rm -rf __pycache__ .ipynb_checkpoints
deactivate  # Exit virtual environment
```

---

## Support and Additional Resources

- **Jupyter Documentation**: https://jupyter.org/
- **Scikit-learn Guide**: https://scikit-learn.org/stable/
- **Pandas Tutorial**: https://pandas.pydata.org/docs/
- **Project GitHub**: Create issues for bugs or feature requests
- **Community Forums**: Stack Overflow, Kaggle Discussions

---

## Summary

You now have a complete, production-quality customer purchase prediction project with:
- Reproducible machine learning pipeline
- Comprehensive documentation
- Version control setup
- Ready-to-deploy models
- Professional GitHub presence

Good luck with your machine learning project!
