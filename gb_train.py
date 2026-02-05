import numpy as np 
import pandas as pd 
import pickle

# preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Regression model
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Metrics 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('insurance.csv')

# print(df[df.duplicated()])
my_df = df.drop_duplicates()     # delete duplicate row

# Feature Engineering
my_df['bmi_group'] = pd.cut(
    df['bmi'],
    bins=[0, 18, 25, 35, 100],
    labels=['under', 'normal', 'over', 'obese']
)

my_df["age_group"] = pd.cut(
    my_df["age"],
    bins=[0, 18, 25, 35, 50, 100],
    labels=["0-18", "19-25", "26-35", "36-50", "51+"]
)

# Target and features
X = my_df.drop('charges', axis=1)
y = my_df['charges']

# Column split
numeric_features = X.select_dtypes(include = ['int64','float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

# Preprocessing
num_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps=[ 
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# GradientBoost model pipeline
final_gb_pipeline = Pipeline(
    steps=[ 
        ('preprocessor', preprocessor),
        ('model',  GradientBoostingRegressor(n_estimators=288, max_depth=20, random_state=42))
    ]
)

final_gb_pipeline.fit(X_train, y_train)
final_y_pred = final_gb_pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, final_y_pred)
rmse = np.sqrt(mean_squared_error(y_test, final_y_pred))
r2 = r2_score(y_test, final_y_pred)

print(f"Mean Absolute Error: {mae : .4f}")
print(f"Root Mean Squared Error: {rmse : .4f}")
print(f"R2 Score: {r2 : .4f}")

# Save model (important)
with open("medical_gb_pipeline.pkl", "wb") as f:
    pickle.dump(final_gb_pipeline, f)

print(f"✅ GradientBoosting pipeline saved as medical_gb_pipeline.pkl")