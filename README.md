# E-Commerce Customer Behaviour and Delivery Analysis

## 1. Project Overview

This project focuses on analysing customer behaviour, delivery performance and customer satisfaction in an e-commerce business using data analytics, machine learning and natural language processing (NLP).

The project uses the Olist Brazilian E-Commerce Public Dataset to identify customer purchasing patterns, investigate delivery delays, analyse customer reviews and predict orders that may receive poor customer ratings.

## 2. Project Objectives

- Identify frequent and occasional customers.
- Analyse customer purchasing and spending habits.
- Identify popular product categories.
- Evaluate delivery performance and identify late deliveries.
- Investigate the relationship between delivery delays and customer ratings.
- Analyse customer reviews using sentiment analysis.
- Predict whether an order may receive a poor customer review.
- Compare different machine learning algorithms.


## Dataset Information

Dataset: Brazilian E-Commerce Public Dataset by Olist

Source:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Original dataset provided by Olist.

License: CC BY-NC-SA 4.0

The dataset is used for academic and non-commercial
analysis purposes.

## 3. Dataset Description

The project uses nine CSV datasets:

| Dataset | Description |
|---|---|
| olist_customers_dataset.csv | Customer information |
| olist_geolocation_dataset.csv | Geographic information |
| olist_order_items_dataset.csv | Order item details |
| olist_order_payments_dataset.csv | Payment information |
| olist_order_reviews_dataset.csv | Customer reviews and ratings |
| olist_orders_dataset.csv | Order and delivery information |
| olist_products_dataset.csv | Product information |
| olist_sellers_dataset.csv | Seller information |
| product_category_name_translation.csv | Product category translations |

## 4. Project Features and Techniques

### Feature 1: Customer Behaviour Analysis

- Identify frequent and occasional customers.
- Analyse customer spending habits.
- Discover popular product categories.
- Segment customers using K-Means clustering.

**Techniques:** Data analysis, exploratory data analysis (EDA), K-Means clustering.

### Feature 2: Delivery Performance Analysis

- Identify late deliveries.
- Compare estimated and actual delivery times.
- Analyse delivery performance by seller and location.
- Identify sellers associated with delivery delays.

**Techniques:** Data analysis and data visualization.

### Feature 3: Customer Satisfaction Analysis

- Analyse customer ratings.
- Investigate the relationship between delivery delays and ratings.
- Classify written reviews into positive, negative and neutral sentiments.
- Compare customer satisfaction across delivery categories.

**Techniques:** Natural Language Processing (NLP) and sentiment analysis.

### Feature 4: Poor Customer Review Prediction

- Predict whether an order may receive a poor customer review.
- Train and compare different machine learning algorithms.
- Identify important factors influencing customer dissatisfaction.

**Algorithms:**
1. Logistic Regression
2. Decision Tree
3. Random Forest

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC and PR-AUC.

## 5. Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- Plotly

## 6. Project Folder Structure

```text
ECOMMERCE_CUSTOMER_BEHAVIOUR_AND_DELIVERY_ANALYSIS/
│
├── data/
│   └── Original CSV datasets
│
├── notebooks/
│
├── results/
│   ├── cleaned_data/
│   ├── customer_segmentation/
│   ├── delivery_analysis/
│   ├── customer_satisfaction/
│   ├── poor_review_prediction/
│   └── eda/
│
├── data_inspection.py
├── data_quality_analysis.py
├── data_cleaning.py
├── investigate_missing_data.py
├── inspect_missing_values.py
├── eda_analysis.py
├── customer_segmentation.py
├── delivery_analysis.py
├── customer_satisfaction_analysis.py
├── poor_review_prediction.py
├── main.py
├── run_all.py
│
└── README.md
```

## 7. Installation and Execution

### Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk plotly
```

### Run the complete project

Open the project folder in Visual Studio Code and execute:

```bash
python run_all.py
```

The script runs the implemented project features and generates the corresponding reports and visualizations inside the results folder.

## 8. Expected Outputs

The project generates:

- Customer behaviour summaries.
- Customer segmentation and cluster profiles.
- Delivery performance reports.
- Customer satisfaction and sentiment analysis reports.
- Poor customer review predictions.
- Machine learning model comparison reports.
- Feature importance analysis.
- Graphs and visualizations for all four features.

## 9. Conclusion

This project demonstrates how data analytics, machine learning and natural language processing can be applied to e-commerce data to understand customer behaviour, evaluate delivery performance and improve customer satisfaction.

By identifying purchasing patterns, analysing delivery delays, understanding customer feedback and predicting poor reviews, the project provides useful insights that can help e-commerce businesses improve their operational efficiency and customer experience.