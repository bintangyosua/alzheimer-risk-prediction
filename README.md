# Alzheimer Disease Classification with XGBoost and KNN

This project aims to classify Alzheimer disease status using XGBoost and KNN machine learning algorithms. The interactive analysis and modeling pipeline is implemented using `marimo`, enabling visual and exploratory steps for understanding data and model performance.

## Project Members

1. [Minuettaro](https://github.com/bintangyosua)
2. ...
3. ...

## Overview

This project covers the following stages:

1. Data Exploration: Includes loading data, checking for null values, and inspecting columns.
2. Correlation Analysis: Creates heatmaps and scatter plots for understanding feature relationships.
3. Model Construction: Splits the dataset into training and testing, normalizes data, and creates models.
4. Model Evaluation: Evaluates different classifiers, including XGBoost, KNN, SVC, Random Forest, Gradient Boosting, and Extra Tree classifiers.
5. Conclusion: Summarizes the model performances and compares accuracy across models.

## Requirements

To run this project, ensure you have the following Python packages installed:

- marimo==0.9.14
- numpy
- pandas
- matplotlib
- seaborn
- altair
- scikit-learn
- xgboost

Install dependencies using:

```bash
pip install marimo numpy pandas matplotlib seaborn altair scikit-learn xgboost
```

## Usage

1. Clone the Repository

```bash
git clone <repository-link>
cd <repository-directory>
```

2. Run the Application

```bash
marimo edit main.py
```

3. Navigate through the analysis:
   - Data Exploration: Review data structure and statistics.
   - Correlation Analysis: View relationships between continuous variables with scatter plots and heatmaps.
   - Model Construction: Choose models and tune parameters such as KNN neighbors interactively.
   - Model Evaluation: Compare the accuracy of different models using bar charts.
   - Conclusion: Read the summarized results and model performance insights.

## Conclusion

In the conclusion, models based on decision tree methods (e.g., Random Forest, XGBoost, Gradient Boosting) tend to perform better, achieving over 80% accuracy, while distance-based models like KNN and SVC show lower accuracy.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project is built with marimo, a Python library designed for data science applications.

## References

```bibtex
@misc{rabie_el_kharoua_2024,
title          = {Alzheimer's Disease Dataset},
url            = {https://www.kaggle.com/dsv/8668279},
DOI            = {10.34740/KAGGLE/DSV/8668279},
publisher      = {Kaggle},
author         = {Rabie El Kharoua},
year           = {2024}}
```

```bibtex
@misc{agrawal_scolnick_2023,
  title        = {marimo - an open-source reactive notebook for Python},
  url          = {https://marimo.io/},
  DOI          = {10.5281/zenodo.12735329},
  publisher    = {Zenodo},
  author       = {Agrawal, Akshay and Scolnick, Myles},
  year         = {2023}
}
```
