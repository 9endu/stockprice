import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and preprocess data (same as before)
df = pd.read_csv('symbols_valid_meta.csv')
df['ETF'] = df['ETF'].map({'Y': 1, 'N': 0})
df['Financial Status'] = df['Financial Status'].fillna('Unknown')
categorical_cols = ['Nasdaq Traded', 'Listing Exchange', 'Market Category', 'Test Issue', 'Financial Status']
df = pd.get_dummies(df, columns=categorical_cols)
cols_to_drop = ['Symbol', 'Security Name', 'CQS Symbol', 'NASDAQ Symbol', 'NextShares']
df = df.drop(cols_to_drop, axis=1)
X = df.drop('ETF', axis=1)
y = df['ETF']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "SVM": SVC(kernel='linear', random_state=42)
}

# Evaluate each model
results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Cross-validation (5-fold)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    # Store results
    results[name] = {
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'CV Mean Accuracy': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Overfitting Score': train_accuracy - test_accuracy  # High = overfitting
    }

# Display results
results_df = pd.DataFrame(results).T
print(results_df.sort_values('Test Accuracy', ascending=False))