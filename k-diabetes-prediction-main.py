#Diabetes prediction using Random Forest model on dataset from National Institute of Diabetes and Digestive and Kidney Diseases - https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data
# Kiran Veerabatheni -  @AppliedAIwithKiran
#--------------------------------------------
# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.linear_model import LogisticRegression  #Logistic Regression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import pickle

#LOAD DATASET
def load_and_prepare_diabetes_data(filepath):
    #Load Data
    df = pd.read_csv(filepath)
    
    for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        column_zero_count = len(df[df[column]==0])
           
    #Handle zeros, first replace zeros with NaN, then replace the NaN with Median vlaue
    columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_processed = df.copy() 
    for column in columns_to_process:
        df_processed[column] = df_processed[column].replace(0, np.nan)
        median_value = df_processed[ df_processed[column]!=0 ][column].median()
        df_processed[column].fillna(median_value, inplace=True) 

    # Visualize the distribution of the target variale
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Outcome', data=df_processed)
    plt.title('Distribution of Diabetes Outcome (1 = Positive, 0 = Negative)')

    # Corealtion Matrix
    plt.figure(figsize=(12,10))
    correlation_matrix = df_processed.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Features')

    #Prepare Features and Target
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']

    #Training and Testing Sets
    X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y )

    #scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 
        
    #Return
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_processed


# MODEL TRAINING AND EVALUATION OF RANDOM FOREST
def train_evaluate_diabetes_random_forest_model(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """
    Trains and evaluates a Random Forest model for diabetes prediction.

    Parameters:
        X_train_scaled (array-like): Scaled training features.
        X_test_scaled (array-like): Scaled testing features.
        y_train (array-like): Training target labels.
        y_test (array-like): Testing target labels.
        feature_names (list): Names of the features (used for feature importance plots).

    Returns:
        best_model: The best-trained Random Forest model.
        results: A dictionary containing evaluation metrics.
    """
    # Initialize the Random Forest model
    model = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Perform grid search for hyperparameter tuning
    print("\nTraining Random Forest model...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Make predictions
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Store results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'best_params': grid_search.best_params_
    }

    # Print classification report
    print("\nClassification Report for Random Forest:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix for Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('visualization\\k_random_forest_confusion_matrix.png')
    plt.show()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest')
    plt.legend(loc='lower right')
    plt.savefig('visualization\\k_random_forest_roc_curve.png')
    plt.show()

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title('Feature Importance (Random Forest)')
    plt.bar(range(X_train_scaled.shape[1]), importances[indices])
    plt.xticks(range(X_train_scaled.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('visualization\\k_random_forest_feature_importance.png')
    plt.show()

    # Save the best model as a Pickle file
    with open('models\\k_random_forest_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    # Save training data also
    with open('models\\k_training_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train_scaled,
            'y_train': y_train,
            'feature_names': feature_names
        }, f)
    return best_model, results

# visualize feature distributions
def visualize_feature_distributions(df):
    plt.figure(figsize=(20, 15))
    for i, column in enumerate(df.columns[:-1], 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=column, hue='Outcome', kde=True, bins=30)
        plt.title(f'Distribution of {column} by Diabetes Outcome')
    plt.tight_layout()
    plt.savefig('visualization\\k_feature_distributions.png')


#CALL THE MAIN METHODS
# Main execution block
if __name__ == "__main__":
    # Dataset path
    filepath = "data\\diabetes.csv"
    
    # Load and prepare the data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_processed = load_and_prepare_diabetes_data(filepath)
    
    # Save the scaler to a pickle file
    with open('models\\k_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Visualize feature distributions
    visualize_feature_distributions(df_processed)
    
    # Get feature names for the feature importance plot
    feature_names = df_processed.drop('Outcome', axis=1).columns.tolist()
    
    # Train and evaluate the Random Forest model
    best_model, results = train_evaluate_diabetes_random_forest_model(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    # Print the final results
    print("\nFinal Model Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    print("\nModel saved as 'models/random_forest_model.pkl'")
    print("Scaler saved as 'models/scaler.pkl'")
    print("Visualization images saved in the visualization directory.")



def _load_and_prepare_diabetes_data(filepath):
    #Load Data
    df = pd.read_csv(filepath)

    #Show Dataset info
    print(f"Dataset Shape (# of rows/columns) {df.shape} ")
    print(f"Dataset desc: {df.describe()}")

    #check missing values
    print(f"\n Missing values: {df.isnull().sum()}")

    #check for zero values in the specific columns: (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
    # and get the count of rows where the column value is zero
    for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        # Count the number of zeros in the column
        column_zero_count = len(df[df[column]==0])
        # Print the result in percentages with 2 decimals
        print(f"{column}: {column_zero_count} zeros ({column_zero_count/len(df)*100:.2f}%)")
    
    #Handle zeros, first replace zeros with NaN, then replace the NaN with Median vlaue
    columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_processed = df.copy() 
    for column in columns_to_process:
        #replace zeros with NaN
        df_processed[column] = df_processed[column].replace(0, np.nan)
        #replace NaN with median
        #calculate meadian of non-zero values
        median_value = df_processed[ df_processed[column]!=0 ][column].median()
        #replace nan with median
        df_processed[column].fillna(median_value, inplace=True) 

    # Visualize the distribution of the target variale
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Outcome', data=df_processed)
    plt.title('Distribution of Diabetes Outcome (1 = Positive, 0 = Negative)')
    plt.savefig('visualization\\k_outcome_distribution.png')

    # Corealtion Matrix
    plt.figure(figsize=(12,10))
    correlation_matrix = df_processed.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Features')
    plt.savefig('visualization\\k_correlation_matrix.png')

    #Prepare Features and Target
    #remove the outcome column
    X = df_processed.drop('Outcome', axis=1)
    #extract the outcome from the data frame
    y = df_processed['Outcome']

    #Training and Testing Sets
    X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y )

    #scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) #Fits the scaler to the training data and transforms the training data.
    X_test_scaled = scaler.transform(X_test) #Transforms the testing data using the scaler fitted on the training data.
    print("\nData is prepared and split into training and testing sets.")
        

    #Return
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_processed


    