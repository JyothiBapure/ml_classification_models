a). Problem Statement

This project implements 6 ML classifiers to Predict whether an individual earns more than $50K per year based on demographic and work-related features 
such as age, education, occupation, and hours worked per week. This prediction is useful for economic studies, policy making, and financial planning

b). Dataset Description

The dataset used in this project is the Adult Income Dataset from the UCI Machine Learning Repository.
Total Instances: ~48,000 records
Target Variable: income
<=50K → 0
>50K → 1

| Feature Name   | Description               |
| -------------- | ------------------------- |
| age            | Age of the individual     |
| workclass      | Employment type           |
| fnlwgt         | Final sampling weight     |
| education      | Highest education level   |
| education-num  | Numerical education level |
| marital-status | Marital status            |
| occupation     | Occupation category       |
| relationship   | Relationship status       |
| race           | Race                      |
| sex            | Gender                    |
| capital-gain   | Capital gains             |
| capital-loss   | Capital losses            |
| hours-per-week | Working hours per week    |
| native-country | Country of origin         |

c). Models Used

The following six machine learning models were implemented and evaluated using the same preprocessing pipeline to ensure fair comparison:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes (Gaussian)
Random Forest Classifier
XGBoost Classifier

Comparison Table with the evaluation metrics calculated for all the 6 models on test dataset

| ML Model Name       | Accuracy   | AUC        | Precision  | Recall     | F1 Score   | MCC        |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.8245     | 0.8509     | 0.7014     | 0.4477     | 0.5466     | 0.4615     |
| Decision Tree       | 0.8093     | 0.7391     | 0.5946     | 0.6061     | 0.6003     | 0.4752     |
| KNN                 | 0.8286     | 0.8492     | 0.6566     | 0.5751     | 0.6132     | 0.5055     |
| Naive Bayes         | 0.8039     | 0.8556     | 0.6723     | 0.3318     | 0.4443     | 0.3729     |
| Random Forest       | 0.8531     | 0.9051     | 0.7257     | 0.6082     | 0.6618     | 0.5725     |
| XGBoost             | 0.8722     | 0.9265     | 0.7800     | 0.6396     | 0.7029     | 0.6275     |

Observations on the performance of each model on the chosen dataset.

| ML Model                  | Observation about Model Performance                                                                                                                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression       | Gives good accuracy and stable results. It works well but misses some high-income cases. Generalizes well without significant overfitting. |
| Decision Tree             | Gives decent accuracy and identifies many high-income cases. Slightly lower AUC compared to other models. Performance remains stable across training and test data.                                                   |
| K-Nearest Neighbors (KNN) | Gives balanced precision and recall with consistent performance across datasets. Distance-based learning works effectively after feature scaling. Shows good generalization ability.                                         |
| Naive Bayes               | Achieves good AUC and precision but lower recall and F1-score. Misses many high-income cases compared to other models. Performance remains stable without overfitting.                                |
| Random Forest (Ensemble)  | Shows strong overall predictive performance with high accuracy and AUC. Effectively captures complex feature interactions. Maintains balanced precision and recall with stable generalization.                                      |
| XGBoost (Ensemble)        | Delivers the best overall performance across most evaluation metrics. Achieves high accuracy, AUC, F1-score, and MCC. Demonstrates strong predictive capability and excellent generalization.                                       |


