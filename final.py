import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import streamlit as st
import matplotlib.pyplot as plt

# Disable warning related to Matplotlib's global figure object
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset in the Pandas DataFrame
credit_card_data = pd.read_csv(r'D:\5th sem\daa\project\creditcard.csv')

# Display first 5 rows of the dataset
credit_card_data.head()

# Check for missing values
credit_card_data.isnull().sum()

# Display distribution of legitimate and fraudulent transactions
class_distribution = credit_card_data['Class'].value_counts()

# Separate data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Display statistical measures of the data
legit_amount_stats = legit.Amount.describe()
fraud_amount_stats = fraud.Amount.describe()

# Compare the values for both transactions
mean_by_class = credit_card_data.groupby('Class').mean()

# Balance the dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Display head and tail of the new dataset
new_dataset.head()
new_dataset.tail()

# Display class distribution in the new dataset
class_distribution_new = new_dataset['Class'].value_counts()

# Prepare features and target variable
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)

# Train Random Forest Classifier Model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, Y_train)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

# Train k-Nearest Neighbors (KNN) Model
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_resampled, Y_resampled)

# Define a list of sampling techniques to compare
sampling_techniques = {
    'No Sampling': (X_train, Y_train),
    'SMOTE': (X_resampled, Y_resampled),
    'Random Under Sampling': RandomUnderSampler(random_state=42).fit_resample(X_train, Y_train),
    'SMOTE + ENN': SMOTEENN(random_state=42).fit_resample(X_train, Y_train)
}

# Train models for each sampling technique
model_accuracies = {}
for technique, (X_train_sampled, Y_train_sampled) in sampling_techniques.items():
    # Train Logistic Regression Model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_sampled, Y_train_sampled)

    # Train Random Forest Classifier Model
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train_sampled, Y_train_sampled)

    # Train k-Nearest Neighbors (KNN) Model
    knn_model = KNeighborsClassifier(n_neighbors=5)  # You can change the number of neighbors as needed
    knn_model.fit(X_train_sampled, Y_train_sampled)

    # Get model accuracies
    logistic_accuracy = accuracy_score(Y_test, logistic_model.predict(X_test))
    random_forest_accuracy = accuracy_score(Y_test, random_forest_model.predict(X_test))
    knn_accuracy = accuracy_score(Y_test, knn_model.predict(X_test))

    model_accuracies[technique] = {
        'Logistic Regression': logistic_accuracy,
        'Random Forest Classifier': random_forest_accuracy,
        'k-Nearest Neighbors (KNN)': knn_accuracy
    }

# Find the technique with the highest overall accuracy
best_technique = max(model_accuracies, key=lambda k: sum(model_accuracies[k][model] for model in model_accuracies[k]))
best_accuracies = model_accuracies[best_technique]

# Find the model with the highest accuracy
best_model = max(best_accuracies, key=best_accuracies.get)

# Streamlit App
st.title("Credit Card Detection Model")

# Change the background
st.markdown(
    """
    <style>
        body {
            background-color: #3498db; /* Change to your desired color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

input_df = st.text_input('Enter All Required Features Values ')
input_df_split = input_df.split(',')

submit = st.button("Submit")
if submit:
    np_df = np.asarray(input_df_split, dtype=np.float64)

    decision_result = None

    if best_model == 'Logistic Regression':
        prediction = logistic_model.predict(np_df.reshape(1, -1))
    elif best_model == 'Random Forest Classifier':
        prediction = random_forest_model.predict(np_df.reshape(1, -1))
    elif best_model == 'k-Nearest Neighbors (KNN)':
        prediction = knn_model.predict(np_df.reshape(1, -1))

    if prediction[0] == 0:
        decision_result = 'Legitimate Transaction'
    else:
        decision_result = 'Fraudulent Transaction'

    if prediction[0] == 1:
        st.markdown(
            """
            <style>
                .stWarning {
                    color: red;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.warning("Warning: This is a Fraudulent Transaction!")

    st.markdown('<div class="result-container">', unsafe_allow_html=True)

    # Display Confusion Matrix
    st.write(f"Confusion Matrix (Best Technique: {best_technique}):")
    if best_model == 'Logistic Regression':
        cm = confusion_matrix(Y_test, logistic_model.predict(X_test))
    elif best_model == 'Random Forest Classifier':
        cm = confusion_matrix(Y_test, random_forest_model.predict(X_test))
    elif best_model == 'k-Nearest Neighbors (KNN)':
        cm = confusion_matrix(Y_test, knn_model.predict(X_test))
    st.write(cm)

    st.write(f'Decision made using: {best_model} (Technique: {best_technique})')
    st.write(f'Result: {decision_result}')

    st.write('Model Accuracies for Best Technique:')
    st.write(f'Logistic Regression: {best_accuracies["Logistic Regression"]}')
    st.write(f'Random Forest Classifier: {best_accuracies["Random Forest Classifier"]}')
    st.write(f'k-Nearest Neighbors (KNN): {best_accuracies["k-Nearest Neighbors (KNN)"]}')

    # Display Charts
    st.bar_chart({
        'Logistic Regression': best_accuracies["Logistic Regression"],
        'Random Forest Classifier': best_accuracies["Random Forest Classifier"],
        'k-Nearest Neighbors (KNN)': best_accuracies["k-Nearest Neighbors (KNN)"]
    })

    st.markdown('</div>', unsafe_allow_html=True)
