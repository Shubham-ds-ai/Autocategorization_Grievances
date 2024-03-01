---

# DARPG Challenge 2024 Submission

## Overview

This project, developed by Shubham Saxena and Veenus Yadav, is a submission for the DARPG Challenge 2024. It aims to address the challenge of auto-categorizing received grievance reports for efficient sharing with the concerned last-mile officers. Utilizing advanced AI/ML techniques, our solution not only categorizes the grievances but also includes a mechanism for the distribution of reports to registered officials and monitoring their status.

## Problem Statement

The DARPG Challenge 2024 called for the development of an AI/ML-driven system for topic clustering/modelling to enable auto-categorisation of received grievance reports. The system is expected to facilitate the sharing of categorized grievances with relevant officials and allow for the monitoring and tracking of these reports.

## Methodology

### Data Preparation
- **Dataset Assessment**: Initial analysis to understand the data structure and content.
- **Language Filtering**: Retention of rows containing English text only.
- **Cleaning and Preprocessing**: Removal of stop words, lemmatization, and cleaning of the text data.
- **Topic Clustering Attempt**: Analysis of common words to form the basis of topic clustering.

### Handling Imbalanced Data
- **Category Consolidation**: Lesser-populated categories were merged into a single 'Others' category to address data imbalance.
- **Separate DataFrames**: Maintenance of a separate DataFrame for the 'Others' category to facilitate specialized processing.

### Model Training and Prediction
- **BERT-large for Main Categories**: Training of the BERT-large model on the cleaned and processed dataset for primary category prediction.
- **BERT-large for Subcategories**: Utilization of another BERT-large model specifically trained on the 'Others' DataFrame for subcategory prediction within the 'Others' category.

## Installation

Instructions on setting up the project environment:

```bash
# Clone the repository
git clone [repository-url]

# Navigate to the project directory
cd [project-directory]

# Install dependencies
pip install -r requirements.txt
```

## Usage

Provide step-by-step instructions on how to run the various components of your project, including data preparation, model training, and prediction.

```bash
# Data preparation
python data_preprocessing/cleaning.py

# Model training
python bert_training/train_bert.py

# Prediction
python prediction/predict.py
```

## Contributors

- Shubham Saxena
- Veenus Yadav

## Acknowledgments

Special thanks to Datasets providers and resource providers.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
