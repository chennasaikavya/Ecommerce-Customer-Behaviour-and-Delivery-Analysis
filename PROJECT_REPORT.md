# E-COMMERCE CUSTOMER BEHAVIOUR AND DELIVERY ANALYSIS

## 1. Introduction

E-commerce has become an important part of modern business. Understanding customer purchasing behaviour, delivery performance and customer satisfaction is essential for improving business operations and customer experience.

This project, titled **E-Commerce Customer Behaviour and Delivery Analysis**, uses data analytics, machine learning and Natural Language Processing (NLP) to analyse e-commerce transactions and customer feedback.

The project uses the Olist Brazilian E-Commerce Public Dataset to investigate customer purchasing patterns, delivery delays, customer ratings and factors influencing poor customer reviews.

## 2. Problem Statement

E-commerce businesses face several challenges, including understanding customer purchasing habits, identifying delivery delays and maintaining customer satisfaction.

Late deliveries and unsatisfactory shopping experiences can negatively affect customer ratings and reviews.

This project aims to address these challenges through data analysis and predictive modelling to generate useful insights that can support better business decisions.

## 3. Project Objectives

The main objectives are:

1. Identify frequent and occasional customers.
2. Analyse customer spending habits.
3. Identify popular product categories.
4. Evaluate delivery performance and identify late deliveries.
5. Analyse the relationship between delivery delays and customer ratings.
6. Perform sentiment analysis on written customer reviews.
7. Predict whether an order may receive a poor customer review.
8. Compare different machine learning algorithms.
9. Identify important factors influencing customer dissatisfaction.

## 4. Dataset Description

The project uses the Olist Brazilian E-Commerce Public Dataset, which consists of nine CSV files.

| Dataset | Description |
|---|---|
| Customers | Customer details and locations |
| Geolocation | Geographic coordinates |
| Order Items | Products included in orders |
| Order Payments | Payment details |
| Order Reviews | Customer ratings and written reviews |
| Orders | Order and delivery information |
| Products | Product details and categories |
| Sellers | Seller information |
| Category Translation | English product category names |

### Initial Dataset Information

The initial dataset inspection identified:

- 99,441 customer records.
- 1,000,163 geolocation records.
- 112,650 order item records.
- 103,886 payment records.
- 99,224 review records.
- 99,441 order records.
- 32,951 product records.
- 3,095 seller records.
- 71 product category translations.

## 5. Data Preprocessing and Cleaning

Data preprocessing was performed before conducting the analysis.

The following activities were included:

- Inspecting dataset structures and column information.
- Identifying missing values.
- Detecting duplicate records.
- Investigating missing data patterns.
- Cleaning datasets and handling missing values.
- Preparing cleaned datasets for further analysis.

The initial inspection identified missing values in the order, product and review datasets. Duplicate records were also identified in the geolocation dataset.

The original datasets were preserved, and cleaned copies were stored in the results folder.

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the characteristics and patterns within the datasets.

The analysis covered:

- Customer purchasing behaviour.
- Customer spending distribution.
- Popular product categories.
- Delivery performance.
- Delivery delays.
- Customer rating distribution.
- Customer review patterns.

Visualizations were generated using Matplotlib and Seaborn to support the interpretation of the results.

## 7. Feature 1: Customer Behaviour Analysis

### 7.1 Customer Frequency Analysis

Customers were classified into frequent and occasional customers based on their purchasing activity.

The analysis showed that occasional customers formed the larger group, while frequent customers represented a smaller proportion.

### 7.2 Customer Spending Analysis

Customer spending habits were examined using purchasing information and order values.

The analysis helped identify differences in spending patterns among customer groups.

### 7.3 Product Category Analysis

Product categories were analysed to identify popular categories and understand purchasing preferences.

### 7.4 Customer Segmentation Using K-Means

K-Means clustering was implemented to group customers based on their purchasing and spending characteristics.

The number of clusters was evaluated using:

- Elbow Method.
- Silhouette Score.

The highest silhouette score was approximately 0.535 at K = 4.

Therefore, four customer clusters were selected.

A total of 93,358 customers were segmented.

The resulting cluster sizes were:

| Cluster | Number of Customers |
|---|---:|
| Cluster 0 | 8,427 |
| Cluster 1 | 45,629 |
| Cluster 2 | 2,801 |
| Cluster 3 | 36,501 |
| **Total** | **93,358** |

Cluster profiles were generated to compare the average purchasing and spending characteristics of each group.

## 8. Feature 2: Delivery Performance Analysis

Delivery performance was analysed by comparing estimated delivery dates with actual delivery dates.

The analysis focused on:

- Actual delivery duration.
- Estimated delivery duration.
- Delivery delays.
- On-time and early deliveries.
- Late deliveries.
- Seller delivery performance.
- Geographic delivery performance.

### 8.1 Delivery Analysis Results

The analysis covered 96,470 delivery records.

Key findings:

- On-time or early deliveries: 88,644.
- Late delivery rate: 8.11%.
- Average actual delivery duration: 12.56 days.
- Average delay: -11.18 days.

The negative average delay indicates that deliveries were earlier than their estimated dates on average, using actual delivery duration minus estimated delivery duration as the delay calculation.

### 8.2 Seller and Location Analysis

Seller delivery performance and geographic delivery patterns were analysed to identify sellers and locations associated with delivery delays.

Visualizations were generated to compare seller performance and delivery delays across locations.

## 9. Feature 3: Customer Satisfaction Analysis

Customer satisfaction was analysed using customer ratings and written reviews.

A total of 99,224 customer reviews were analysed.

### 9.1 Customer Rating Analysis

Customer ratings were examined to understand satisfaction levels.

The analysis included:

- Rating distribution.
- Ratings by delivery status.
- Ratings by delivery delay category.
- Relationship between delivery delays and customer ratings.


### 9.2 Sentiment Analysis

Natural Language Processing (NLP) was used to classify written customer reviews into three sentiment categories:

1. Positive.
2. Negative.
3. Neutral.

A total of 40,779 reviews were classified.

The sentiment analysis results were:

| Sentiment | Number of Reviews | Percentage |
|---|---:|---:|
| Positive | 23,372 | 57.31% |
| Neutral | 13,266 | 32.53% |
| Negative | 4,141 | 10.15% |

The results show that positive reviews represent the largest proportion of classified customer feedback, while negative reviews account for approximately 10.15%.

This analysis provides insights into customer opinions and overall satisfaction.

### 9.3 Delivery Delays and Customer Satisfaction

The relationship between delivery delays and customer
ratings was investigated to understand how delivery
performance influences customer satisfaction.

The analysis produced the following results:

| Delivery Delay Category | Total Reviews | Average Rating |
|---|---:|---:|
| Early / On time | 88653 | 4.29 |
| 1-3 Days Late | 2651 | 3.77 |
| 4-7 Days Late | 1777 | 2.32 |
| More than 7 Days Late | 3272 | 1.73 |

**Key Findings:**

- Early or on-time deliveries received the highest
  average rating of 4.29.
- Deliveries delayed by 1-3 days received an
  average rating of 3.77.
- Deliveries delayed by 4-7 days received an
  average rating of 2.32.
- Deliveries delayed by more than 7 days received
  the lowest average rating of 1.73.

The results indicate a clear downward trend in
customer ratings as delivery delays increase.

**Conclusion:**

Timely delivery plays an important role in
customer satisfaction. E-commerce businesses
should improve delivery planning and logistics
management to reduce delays and improve
customer experience.
## 10. Feature 4: Poor Customer Review Prediction

Machine learning was implemented to predict whether an order may receive a poor customer review.

The prediction model used order, payment, product and delivery-related features.

Three machine learning algorithms were evaluated:

1. Logistic Regression.
2. Decision Tree.
3. Random Forest.

### 10.1 Model Evaluation

The models were compared using:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.

The model comparison results are presented below:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 85.12% | 42.94% | 50.31% | 46.33% | 75.29% | 47.84% |
| Logistic Regression | 81.96% | 36.06% | 53.41% | 43.05% | 74.87% | 43.11% |
| Decision Tree | 82.01% | 36.13% | 53.21% | 43.03% | 72.55% | 44.70% |

**Key Findings:**

- Random Forest achieved the highest accuracy of 85.12%.
- Random Forest achieved the highest F1 Score of 46.33%.
- Logistic Regression achieved the highest recall of 53.41%.
- Random Forest achieved the highest ROC AUC of 75.29%.
- Random Forest provided the strongest overall performance among the three evaluated models.

Based on the evaluation results, Random Forest was selected as the best-performing model for predicting poor customer reviews.

### 10.2 Model Selection and Interpretation.

The three machine learning models were compared to identify the most suitable model for predicting poor customer reviews.

**Model Comparison Findings:**

- Random Forest demonstrated the strongest overall performance, achieving the highest accuracy, precision, F1-score, ROC AUC and PR AUC.
- Logistic Regression achieved the highest recall of 53.41%, indicating that it identified a slightly larger proportion of actual poor-review cases.
- Decision Tree achieved an accuracy of 82.01%, but its overall performance was lower than Random Forest.

**Model Selection:**

Random Forest was selected as the final model because it provided the best overall balance across the evaluation metrics.

Although Logistic Regression achieved slightly higher recall, Random Forest provided better precision and F1-score, making it the preferred model for this project.

### 10.3 Feature Importance Analysis

Feature importance analysis was performed to identify the factors contributing most to poor review predictions.

The most influential features included:

| Feature | Importance |
|---|---:|
| Delay Days | 0.2573 |
| Actual Delivery Days | 0.1952 |
| Late Delivery | 0.1542 |
| Item Count | 0.0655 |
| Total Freight | 0.0514 |

Delivery-related factors were among the most influential features in predicting poor customer reviews.

## 11. Results and Discussion

The project successfully implemented four major analytical features.

The customer behaviour analysis identified purchasing patterns and grouped 93,358 customers into four clusters.

The delivery performance analysis examined 96,470 delivery records and identified an 8.11% late delivery rate.

The customer satisfaction analysis examined 99,224 reviews using rating analysis and sentiment classification.

The poor customer review prediction component evaluated three machine learning algorithms. Random Forest achieved the highest accuracy of approximately 85.12%.

Feature importance analysis also highlighted the importance of delivery-related factors in predicting poor customer reviews.

Together, these results demonstrate how data analytics and machine learning can provide useful insights into e-commerce operations and customer satisfaction.

## 12. Limitations

Some limitations of the project include:

- The analysis is based on historical e-commerce data.
- Missing values and duplicate records require appropriate preprocessing.
- Customer purchasing behaviour may change over time.
- Sentiment classification may not capture every aspect of customer opinions.
- Prediction performance depends on the available features and dataset quality.

## 13. Future Enhancements

The project could be improved through:

- Real-time customer behaviour monitoring.
- More advanced customer segmentation techniques.
- Improved delivery delay prediction.
- Advanced NLP models for review classification.
- Hyperparameter tuning to improve machine learning performance.
- Interactive dashboards for business users.
- Deployment as a web-based analytical application.

## 14. Conclusion

The E-Commerce Customer Behaviour and Delivery Analysis
project successfully demonstrates the application of data
analytics, machine learning and Natural Language Processing
to real-world e-commerce data.

The project produced the following key findings:

1. Customer Behaviour Analysis:
   Customer purchasing patterns and spending behaviour
   were analysed to understand different customer groups.

2. Customer Segmentation:
   A total of 93,358 customers were segmented into
   four clusters using machine learning techniques.

3. Delivery Performance Analysis:
   A total of 96,470 delivery records were analysed.
   The late delivery rate was approximately 8.11%,
   highlighting the importance of delivery management.

4. Customer Satisfaction Analysis:
   Sentiment analysis identified 57.31% positive,
   32.53% neutral and 10.15% negative reviews
   among the classified review records.

5. Poor Customer Review Prediction:
   Three machine learning models were evaluated.
   Random Forest achieved the highest accuracy
   of approximately 85.12%.

Overall, the project demonstrates how data-driven
insights can help e-commerce businesses understand
customer behaviour, improve delivery performance,
monitor customer satisfaction and support better
business decisions.

The project also provides opportunities for future
improvements through advanced NLP techniques,
model optimisation and interactive dashboards.

---

**Project Title:** E-Commerce Customer Behaviour and Delivery Analysis

**Dataset:** Olist Brazilian E-Commerce Public Dataset

**Programming Language:** Python

**Development Environment:** Visual Studio Code