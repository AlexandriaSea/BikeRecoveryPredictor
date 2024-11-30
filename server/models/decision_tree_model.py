import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/AlexandriaSea/BikeRecoveryDataset/main/Bicycle_Thefts_Open_Data.csv')

# Create binary target for recovered status based on 'STATUS' column
df['IS_RECOVERED'] = df['STATUS'].apply(lambda x: 1 if x == 'RECOVERED' else 0)

# Check class distribution
class_distribution = df['IS_RECOVERED'].value_counts()
print("Class distribution for IS_RECOVERED before dropping:\n", class_distribution)

# Drop unnecessary columns
df_cleaned = df.drop(columns=['OBJECTID', 'EVENT_UNIQUE_ID', 'OCC_DATE', 'REPORT_DATE', 
                              'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140', 'STATUS', 'LONG_WGS84', 'LAT_WGS84', 'x', 'y'])

# Drop rows where 'HOOD_158' has the value 'NSA'
df_cleaned = df_cleaned[df_cleaned['HOOD_158'] != 'NSA']

# Check initial missing values
missing_data = df_cleaned.isnull().sum()
print("Initial missing values:\n", missing_data)
print(f"Initial dataset size: {len(df_cleaned)}")

# Define categorical and numerical columns
categorical_columns = ['LOCATION_TYPE', 'PREMISES_TYPE', 'BIKE_MAKE', 'BIKE_TYPE', 'BIKE_COLOUR', 'BIKE_MODEL',
                       'PRIMARY_OFFENCE', 'OCC_MONTH', 'REPORT_MONTH', 'OCC_DOW', 'REPORT_DOW', 'DIVISION']
numerical_columns = list(df_cleaned.select_dtypes(include=['int64', 'float64']).columns)
numerical_columns.remove('IS_RECOVERED')

# Define the preprocessing pipeline with imputation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler())
        ]), numerical_columns),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=True, handle_unknown='ignore'))
        ]), categorical_columns)
    ],
    remainder='passthrough'
)

# Define features and target
X = df_cleaned.drop(columns=['IS_RECOVERED'])
y_recovered = df_cleaned['IS_RECOVERED']

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y_recovered.value_counts()}")

# Stratified Split to ensure balanced class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y_recovered, test_size=0.3, random_state=42, stratify=y_recovered)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Calculate class weights
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(y_recovered),
                                   y=y_recovered)
weight_dict = dict(zip(np.unique(y_recovered), class_weights))

# Create the final pipeline with Decision Tree Classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        max_depth=5,  # Limiting tree depth to avoid overfitting
        random_state=42,
        class_weight=weight_dict
    ))
])

# Function to train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test, target_name):
    # Transform features first
    X_train_transformed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
    
    # Fit the classifier
    pipeline.named_steps['classifier'].fit(X_train_transformed, y_train)
    
    # Save the trained model
    with open('pkl/decision_tree_model.pkl', 'wb') as file:
        pickle.dump(pipeline, file)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Display results
    print(f"Evaluation for {target_name} Prediction:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC Score: {roc_auc}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\n" + "-"*50 + "\n")
    
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Recovered', 'Recovered'], 
                yticklabels=['Not Recovered', 'Recovered'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('plot/confusion_matrix_dt.png')
    plt.show()
    

# Train and evaluate for 'IS_RECOVERED' prediction
try:
    print("Training and Evaluation for 'IS_RECOVERED' Prediction")
    train_and_evaluate(X_train, X_test, y_train, y_test, "Bike Recovered")
except Exception as e:
    print(f"Error during training and evaluation: {str(e)}")
