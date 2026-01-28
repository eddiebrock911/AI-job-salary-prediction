import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    print("Loading data...")
    df = pd.read_csv('ai_job_dataset.csv')
    
    # Preprocessing
    drop_cols = ['job_id', 'salary_currency', 'posting_date', 'application_deadline', 'job_description_length']
    df_clean = df.drop(columns=drop_cols, errors='ignore')
    
    # Fix: Create separate LabelEncoder for each categorical column
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        
    X = df_clean.drop('salary_usd', axis=1)
    y = df_clean['salary_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model: {name}")
        print(f"MAE: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
    
    # Save the best model
    print(f"\n{'='*50}")
    print(f"Saving best model: {best_name} (R2: {best_r2:.4f})")
    joblib.dump(best_model, 'salary_predictor_model.pkl')
    print("Model saved successfully to 'salary_predictor_model.pkl'")
    print(f"{'='*50}")

except Exception as e:
    print(f"Error: {e}")
