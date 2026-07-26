# Customer Purchase Prediction Using Classification Algorithms

A machine learning project that predicts whether customers will make purchases using multiple classification algorithms. This repository contains a complete workflow from data exploration to model evaluation and comparison.

## Project Overview

This project implements and compares four classification algorithms to predict customer purchase behavior. The analysis provides actionable insights into which factors influence purchase decisions and which models perform best for this task.

## Problem Statement

Businesses need to identify high-intent customers to optimize marketing spend and improve conversion rates. Manual analysis of customer data is time-consuming and prone to bias. This project develops an automated classification system to identify customers likely to make purchases based on their behavioral and demographic features.

## Objectives

- Build and train multiple classification models
- Compare model performance using standard evaluation metrics
- Identify key features influencing purchase decisions
- Provide insights for targeted marketing strategies
- Create a reproducible, production-ready workflow

## Repository Structure

```
Customer-Purchase-Prediction/
├── Customer_Purchase_Prediction.ipynb  # Main Jupyter notebook
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
├── LICENSE                              # MIT License
├── report.md                            # Detailed analysis report
├── images/                              # Directory for visualizations
└── data/
    └── customer_data.csv               # Dataset
```

## Dataset Description

The dataset contains customer behavioral and demographic features used to predict purchase decisions. The typical structure includes:

- **Numerical Features**: Metrics such as age, income, browsing time, click rates
- **Categorical Features**: Features such as customer segment, device type, geographic region
- **Target Variable**: Binary indicator of purchase (1: Purchased, 0: Did Not Purchase)

Expected dataset dimensions: Multiple rows of customer records with features and target variable.

### Data Characteristics

- No assumption on specific dataset size
- Handles missing values through imputation
- Supports both numerical and categorical features
- Works with binary classification targets

## Machine Learning Workflow

1. **Data Loading & Exploration**: Load dataset and perform initial exploration
2. **Exploratory Data Analysis**: Understand distributions, correlations, and patterns
3. **Data Preprocessing**: Handle missing values, encode categories, scale features
4. **Model Training**: Train four classification algorithms
5. **Model Evaluation**: Evaluate using accuracy, precision, recall, F1-score
6. **Model Comparison**: Compare performance across metrics
7. **Feature Importance**: Identify most influential features
8. **Conclusion**: Summarize findings and recommendations

## Models Used

The project implements and compares four classification algorithms:

### 1. Logistic Regression
- Linear model with probabilistic output
- Best for: Interpretability and fast inference
- Advantages: Simple, fast, provides probability scores
- Disadvantages: Limited to linear relationships

### 2. Decision Tree
- Tree-based model with explicit decision rules
- Best for: Interpretability and handling non-linear patterns
- Advantages: Easy to understand and visualize
- Disadvantages: Prone to overfitting

### 3. Random Forest
- Ensemble of decision trees with bagging
- Best for: Balanced accuracy and feature importance
- Advantages: Reduces overfitting, provides feature importance
- Disadvantages: Less interpretable than single tree

### 4. K-Nearest Neighbors (KNN)
- Instance-based learning algorithm
- Best for: Non-linear patterns with small datasets
- Advantages: Simple and intuitive
- Disadvantages: Computationally expensive, sensitive to scaling

## Evaluation Metrics

The project evaluates models using the following metrics:

- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/False positives and negatives
- **ROC Curve**: Trade-off between true positive and false positive rates

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Customer-Purchase-Prediction.git
cd Customer-Purchase-Prediction
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## How to Run

### Option 1: Jupyter Notebook
```bash
jupyter notebook Customer_Purchase_Prediction.ipynb
```
Then execute cells in order from top to bottom.

### Option 2: Command Line
```bash
jupyter nbconvert --to notebook --execute Customer_Purchase_Prediction.ipynb
```

### Requirements
1. Ensure `data/customer_data.csv` exists with appropriate structure
2. All dependencies from `requirements.txt` must be installed
3. Run notebook cells sequentially for proper execution

## Project Screenshots

After running the notebook, you'll generate the following visualizations:

- **Target Distribution Chart**: Bar plot showing class balance
- **Feature Histograms**: Distribution of numerical features
- **Boxplots**: Outlier detection for numerical features
- **Correlation Heatmap**: Feature correlation analysis
- **Model Comparison Chart**: Performance comparison bar plot
- **Feature Importance Chart**: Most influential features ranking
- **Feature Coefficients**: Logistic regression coefficients visualization

These visualizations are saved during notebook execution and can be manually saved to the `images/` directory.

## Dataset Format

The `data/customer_data.csv` file should have the following structure:

```
feature1,feature2,feature3,...,featureN,Purchase
value1,value2,value3,...,valueN,1
value1,value2,value3,...,valueN,0
...
```

The target column (Purchase) should be the last column in the dataset.

## Future Improvements

1. **Advanced Feature Engineering**
   - Interaction features
   - Polynomial features
   - Domain-specific derived features

2. **Hyperparameter Optimization**
   - Grid search for optimal parameters
   - Random search for complex spaces
   - Bayesian optimization for efficiency

3. **Model Improvements**
   - Gradient boosting models (XGBoost, LightGBM)
   - Neural networks for complex patterns
   - Ensemble voting classifiers

4. **Deployment**
   - REST API for model serving
   - Docker containerization
   - Cloud deployment (AWS, GCP, Azure)

5. **Monitoring**
   - Model performance tracking
   - Data drift detection
   - Automated retraining pipelines

6. **Explainability**
   - SHAP values for feature attribution
   - LIME for local interpretability
   - Model-agnostic analysis tools

## Code Quality Standards

This project follows PEP 8 guidelines:
- Clear variable naming
- Modular function design
- Minimal code duplication
- Comprehensive comments for complex sections
- Type hints for function parameters

## Testing

To validate the code works correctly:
```bash
# Load and inspect dataset
python -c "import pandas as pd; df = pd.read_csv('data/customer_data.csv'); print(df.shape)"

# Check dependencies
python -c "import sklearn, pandas, numpy; print('All imports successful')"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a pull request

### Guidelines
- Follow PEP 8 code style
- Add comments for complex logic
- Update documentation as needed
- Test code before submitting PR

## Author

Created as a portfolio project demonstrating machine learning workflow and best practices.

## Acknowledgments

- Scikit-learn for ML algorithms
- Pandas for data manipulation
- Matplotlib and Seaborn for visualizations
- Jupyter for interactive notebook environment

## Support

For questions or issues:
1. Check existing GitHub issues
2. Review the report.md for detailed analysis
3. Consult scikit-learn documentation
4. Create a new issue with detailed description

## Changelog

### Version 1.0
- Initial release with four classification models
- Comprehensive EDA and preprocessing
- Full evaluation metrics and comparison
- Feature importance analysis
