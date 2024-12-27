import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Generate sample house data
np.random.seed(42)
n_samples = 200

# Generate features
square_feet = np.random.normal(2000, 500, n_samples)
num_bedrooms = np.random.randint(1, 6, n_samples)
num_bathrooms = np.random.randint(1, 4, n_samples)
lot_size = np.random.normal(5000, 1500, n_samples)
age = np.random.randint(0, 50, n_samples)
garage_size = np.random.randint(0, 4, n_samples)

# Create some correlations and add noise
base_price = (
    square_feet * 150 +  # $150 per sq ft
    num_bedrooms * 25000 +  # $25k per bedroom
    num_bathrooms * 15000 +  # $15k per bathroom
    lot_size * 10 +  # $10 per sq ft of lot
    garage_size * 10000 -  # $10k per garage space
    age * 1000 +  # Depreciation $1k per year
    np.random.normal(0, 50000, n_samples)  # Random noise
)

# Create DataFrame
data = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': num_bedrooms,
    'bathrooms': num_bathrooms,
    'lot_size': lot_size,
    'age': age,
    'garage_size': garage_size,
    'price': base_price
})

print("Step 1: Data Overview")
print("\nSample of house data:")
print(data.head())
print("\nData Description:")
print(data.describe())

# Split features and target
X = data.drop('price', axis=1)
y = data['price']

# Get feature names before splitting
feature_names = X.columns.tolist()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nStep 2: Data Preparation")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names
)

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, 
                              cv=5, scoring='r2')
    
    print(f"\n{model_name} Results:")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Average CV R² score: {cv_scores.mean():.3f}")
    
    if isinstance(model, LinearRegression):
        # Print feature importance for Linear Regression
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        })
        print("\nFeature Coefficients (standardized):")
        print(importance.sort_values(by='Coefficient', ascending=False))
    
    return model

print("\nStep 3: Model Training and Evaluation")

# 1. Linear Regression
lr_model = evaluate_model(
    LinearRegression(),
    X_train_scaled, X_test_scaled, y_train, y_test,
    "Linear Regression"
)

# 2. Ridge Regression (L2 regularization)
ridge_model = evaluate_model(
    Ridge(alpha=1.0),
    X_train_scaled, X_test_scaled, y_train, y_test,
    "Ridge Regression"
)

# 3. Lasso Regression (L1 regularization)
lasso_model = evaluate_model(
    Lasso(alpha=1.0),
    X_train_scaled, X_test_scaled, y_train, y_test,
    "Lasso Regression"
)

print("\nStep 4: Prediction Example")
# Create sample houses for prediction
sample_houses = pd.DataFrame({
    'square_feet': [1500, 2500, 3500],
    'bedrooms': [2, 3, 4],
    'bathrooms': [1, 2, 3],
    'lot_size': [4000, 6000, 8000],
    'age': [20, 10, 0],
    'garage_size': [1, 2, 3]
})

print("\nPredicting prices for sample houses:")
print("\nSample house features:")
print(sample_houses)

# Scale the sample houses
sample_houses_scaled = pd.DataFrame(
    scaler.transform(sample_houses),
    columns=feature_names
)

# Make predictions using all models
print("\nPredicted prices:")
models = {
    'Linear Regression': lr_model,
    'Ridge Regression': ridge_model,
    'Lasso Regression': lasso_model
}

for name, model in models.items():
    predictions = model.predict(sample_houses_scaled)
    print(f"\n{name} predictions:")
    for i, pred in enumerate(predictions):
        print(f"House {i+1}: ${pred:,.2f}")

# Print feature correlations with price
print("\nFeature Correlations with Price:")
correlations = data.corr()['price'].sort_values(ascending=False, key=lambda x: abs(x))
print(correlations)