import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# streamlit run app.py

# Set page config
st.set_page_config(page_title="AI Job Salary Predictor", layout="wide", page_icon="🤖")

# Title and description
st.title("🤖 AI Job Salary Predictor")
st.markdown("Enter job details below to predict the estimated salary in USD.")

# Paths
#  MODEL_URL = "https://drive.google.com/uc?export=download&id=1rtjMiGmBGKmeVtB05HfJolnnivcJCyGy"
DATA_PATH = "ai_job_dataset.csv"
MODEL_ID = "1rtjMiGmBGKmeVtB05HfJolnnivcJCyGy"
MODEL_PATH = f"https://drive.google.com/uc?id={MODEL_ID}"

@st.cache_data
def load_data_and_encoders():
    """Load dataset and create encoders for categorical columns"""
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}")
        return None, None, None
    
    df = pd.read_csv(DATA_PATH)
    
    # Columns to drop (matching training preprocessing)
    drop_cols = ['job_id', 'salary_currency', 'posting_date', 'application_deadline', 
                 'job_description_length', 'salary_usd']
    
    # Create feature set
    X_raw = df.drop(columns=drop_cols, errors='ignore')
    
    # Identify categorical columns
    categorical_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
    
    # Create encoders
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_raw[col].astype(str))
        encoders[col] = le
    
    # Get feature statistics for input validation
    feature_stats = {
        col: {
            'min': float(X_raw[col].min()),
            'max': float(X_raw[col].max()),
            'mean': float(X_raw[col].mean()),
            'median': float(X_raw[col].median())
        }
        for col in X_raw.select_dtypes(include=['number']).columns
    }
    
    return X_raw.columns.tolist(), encoders, feature_stats

@st.cache_resource
def load_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

# Load resources
feature_columns, encoders, feature_stats = load_data_and_encoders()
model = load_model()

if feature_columns is not None and model is not None:
    # Create Input Form
    st.sidebar.header("📝 Input Features")
    st.sidebar.markdown("Fill in all fields to predict salary")
    
    user_input = {}
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(feature_columns):
        container = col1 if i % 2 == 0 else col2
        
        if col in encoders:
            # Categorical feature - Dropdown
            options = sorted(list(encoders[col].classes_))
            # Format column name for display
            display_name = col.replace('_', ' ').title()
            user_input[col] = container.selectbox(
                f"{display_name}", 
                options=options, 
                key=col,
                help=f"Select the {display_name.lower()}"
            )
        else:
            # Numerical feature - Number Input
            stats = feature_stats[col]
            display_name = col.replace('_', ' ').title()
            user_input[col] = container.number_input(
                f"{display_name}", 
                min_value=stats['min'], 
                max_value=stats['max'], 
                value=stats['median'],  # Use median as default (more robust than mean)
                key=col,
                help=f"Range: {stats['min']:.2f} - {stats['max']:.2f}"
            )

    # Add some spacing
    st.markdown("---")
    
    # Prediction Button
    col_button, col_space = st.columns([1, 3])
    with col_button:
        predict_button = st.button("🔮 Predict Salary", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Calculating salary prediction..."):
            try:
                # Create DataFrame with correct column order
                input_df = pd.DataFrame([user_input], columns=feature_columns)
                
                # Encode categorical features
                for col, le in encoders.items():
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except ValueError as e:
                        st.error(f"❌ Invalid value for {col}: '{user_input[col]}' not found in training data")
                        st.stop()
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display result with styling
                st.success("### ✅ Prediction Complete!")
                
                # Create metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col2:
                    st.metric(
                        label="Estimated Annual Salary",
                        value=f"${prediction:,.0f}",
                        help="Predicted salary in USD"
                    )
                    st.metric(
                        label="Estimated Monthly Salary",
                        value=f"${prediction/12:,.0f}",
                        help="Predicted salary in USD"
                    )
                    st.metric(
                        label="Estimated Weekly Salary",
                        value=f"${prediction/52:,.0f}",
                        help="Predicted salary in USD"
                    )
                    st.metric(
                        label="Estimated Daily Salary",
                        value=f"${prediction/260:,.0f}",
                        help="Predicted salary in USD"
                    )
                    st.metric(
                        label="Estimated Hourly Salary",
                        value=f"${prediction/260/40:,.0f}",
                        help="Predicted salary in USD"
                    )
                   
                # Optional: Show confidence interval or additional info
                st.info("💡 **Note**: This is an estimate based on historical data. Actual salaries may vary based on negotiation, location, and market conditions.")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                with st.expander("🔍 Show Error Details"):
                    st.code(str(e))
                st.info("Please ensure all inputs are valid and match the training data format.")

else:
    st.warning("⚠️ Please ensure 'ai_job_dataset.csv' and 'salary_predictor_model.pkl' are in the current directory.")
    st.info("📁 Required files:\n- `ai_job_dataset.csv` - Training dataset\n- `salary_predictor_model.pkl` - Trained model")

# Add footer
st.markdown("---")
st.markdown("Made with Ankit ❤️")