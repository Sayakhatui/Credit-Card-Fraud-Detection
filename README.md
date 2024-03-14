# Credit-Card-Fraud-Detection
Supervised learning Based credit Card Fraud Detection and deployed on streamlit
##  IMPORTANCE TO DETECTING FRAUDULENT TRANSACTIONS
 Credit Card Fraud Detection is a critical aspect of financial security in the modern world. It involves identifying and preventing unauthorised or fraudulent transactions made using credit cards. With the increasing reliance on digital payments, the risk of fraudulent activities has also escalated. Detecting these activities in real-time is crucial to protect both financial institutions and customers from potential losses. This presentation will delve into various models and techniques used to effectively identify and prevent credit card fraud.  
 
## DATASET CHARACTERISTICS 
Credit Card Fraud Detection from Kaggle https://www.kaggle.com/datasets/kartik2112/fraud-detection: This dataset contains transactions made by credit cards in September 2013 by European cardholders. It's a highly imbalanced dataset, meaning there are many more legitimate transactions than fraudulent ones (around 0.17% of transactions are fraudulent). The features are anonymized for security reasons, but they include information like transaction amount, time since the previous transaction, and various other numerical features.

![image](https://github.com/Sayakhatui/Credit-Card-Fraud-Detection/assets/150340995/c6759a7e-850d-4bb9-8dae-de8d957233e9)
![image](https://github.com/Sayakhatui/Credit-Card-Fraud-Detection/assets/150340995/27609ae6-2f74-4875-8414-f6f2dc1342ba)

## PREPROCESSING TECHNIQUES
Preprocessing techniques are essential for handling imbalanced datasets. These methods help balance the class distribution, which is crucial for training accurate fraud detection models
## Undersampling-
• Description: Reduces instances in the majority class to balance the dataset.
• Advantages: Faster training, reduced computational resources.

## SMOTE (Synthetic Minority Over-sampling Technique):
Description: Generates synthetic samples for the minority class using interpolation.
• Considerations: May lead to loss of information.

## Oversampling
Advantages: Addresses overfitting, creates realistic synthetic samples.
Description: Increases instances in the minority class to balance the dataset.
Considerations: May introduce noise if not applied appropriately.
Advantages: Helps prevent loss of information in the minority class.
Considerations: Careful implementation needed to avoid overfitting.


## DATA MODELLING (ML TECHNIQUES AND MODELS/CLASSIFIER)

Logistic Regression: Logistic Regression is a simple yet effective classifier used for binary and multiclass classification tasks. It models the probability of a data point belonging to a particular class and is widely used in applications like medical diagnosis and spam email detection.

K-Nearest Neighbors (KNN): KNN is a non-parametric classifier that assigns a class label to a data point based on the classes of its k-nearest neighbors in the feature space. It's intuitive and easy to understand, making it suitable for applications such as recommendation systems and anomaly detection.

Random Forest: Random Forest is an ensemble classifier that combines multiple decision trees. It reduces overfitting by considering random subsets of data and features, offering high accuracy and robustness. It's applicable in finance, healthcare, and image recognition.
### MODEL ACCURACIES AND COMPARISON

## K-NN CLASSIFIER 
a) k-Neighbors works even efficiently for this imbalanced datasets.
b) It takes around 3-5 minutes for training.
c) Maximum Accuracy of 99.967466 %and Macro Average of F1-Score of 1.00 acheived with Oversampling Techniques.
FOR DETAILS ON MODEL TRAINIG AND ACCURACY COMPARISONS
```
KNeighbors_model.ipynb
```
## RANDOM FOREST CLASSIFIER -https://github.com/Sayakhatui/Credit-Card-Fraud-Detection/blob/main/random_forest_model.ipynb
a) Undersampling doesn't work efficiently for Large majority class datasets as it ignore many valuable tuples. But, can be efficient for small majority class datasets
b) RandomForest works even efficiently for this imbalanced datasets.
c) RandomForest takes around 10-15 minutes for training.
d) Maximum Accuracy of 99.996483% and macro-average of F1-Score of 1.00 acheived with Oversampling technique.
FOR DETAILS ON MODEL TRAINIG AND ACCURACY COMPARISONS
```
random_forest_model.ipynb
```
## LOGISTIC REGRESSION
a) Logistic Regression doesn't work efficiently for this imbalanced datasets.
b) It takes around 1-2 minutes for training.
c) Maximum Accuracy of 99.912222% and Macro Average of F1-Score of 0.85 acheived with StandardScaled datasets.
FOR DETAILS ON MODEL TRAINIG AND ACCURACY COMPARISONS
```
LOGISTIC_REGRESSSION.ipynb
```
## DEPLOYMENT AND RUNNING 
   ![image](https://github.com/Sayakhatui/Credit-Card-Fraud-Detection/assets/150340995/1cdadd0f-8284-4295-b32f-d3522d539332)
   ![image](https://github.com/Sayakhatui/Credit-Card-Fraud-Detection/assets/150340995/9575a9cf-3d87-400c-9994-829e152e7305)

TO RUN THE COMPLETE MODEL :-
1. DOWNLOAD THE DATASET -https://www.kaggle.com/datasets/kartik2112/fraud-detection
2. DOWNLOAD THE final.py and then run the following commnand in the terminal in the directory present 
```
streamlit run final.py
```
3. now add the row of a single transaction except the label part
4. HIT SUBMIT
