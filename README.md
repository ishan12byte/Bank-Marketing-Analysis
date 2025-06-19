## Overview
This repository contains the code and documentation for analyzing and predicting the success of marketing campaigns using the UCI Bank Marketing dataset. The project focuses on data preprocessing, exploratory analysis, feature engineering, and machine learning pipeline development.

## Features

### 1. Data Loading and Cleaning
- Fetched data directly from the UCI ML repository.
- Cleaned and encoded categorical variables.
- Scaled and normalized numerical features.
- Handled missing values effectively.

### 2. Exploratory Data Analysis (EDA)
- Demographic analysis of age, job, and marital status distributions.
- Correlation heatmap for numerical features.
- Campaign outcome trends by day, month, and duration.

### 3. Feature Engineering
- Created new features such as `balance_duration_ratio` and `contacts_per_day`.
- Segmented data into bins for easier interpretation.
- Applied SMOTE for class imbalance resolution.

### 4. Modeling and Evaluation
- Built a Random Forest Classifier for campaign success prediction.
- Evaluated the model using classification reports and confusion matrix visualization.

## Technologies and Tools
- **Programming Language:** Python
- **Libraries:**
  - Data Handling: Pandas
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
  - Class Imbalance Handling: Imbalanced-learn

## Results
### Visualizations
- Heatmaps, distribution plots, scatterplots, and success rate analysis provide detailed insights.

### Model Performance
- Random Forest Classifier achieved strong performance metrics.
- Confusion matrix plots visualize the model’s accuracy and errors.

## Getting Started

### Prerequisites
Make sure you have Python 3.7+ installed. Install the required packages by running:

```bash
pip install -r requirements.txt
```

### Running the Project
1. Clone the repository:

```bash
git clone https://github.com/ishan12byte/Bank-Marketing-Analysis.git
```

2. Navigate to the project directory:

```bash
cd Bank-Marketing-Analysis
```

3. Run the analysis script:

```bash
python analysis.py

```

Or, explore the project interactively in a Jupyter Notebook.

### File Structure
- `analysis.py`: Main script for data preprocessing, EDA, and modeling.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Acknowledgments
- Data sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or suggestions, feel free to reach out via [ishangupta.cpu@gmail.com / 8728025749].
