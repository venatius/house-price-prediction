import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import os

# Load training and test data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Pick important features
top_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
    '1stFlrSF', 'ExterQual', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
    'KitchenQual', 'GarageFinish', 'YearRemodAdd', 'BsmtQual', 'Foundation',
    'MasVnrArea', 'Fireplaces', 'GarageYrBlt', 'BsmtFinType1'
]

# Keep only selected features
train = train[top_features + ['SalePrice']]
test = test[top_features]

# Fill missing values
num_cols = train.select_dtypes(include=[np.number]).columns.drop('SalePrice')
cat_cols = train.select_dtypes(include=['object']).columns

for col in num_cols:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())

for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(train[col].mode()[0])

# Map quality ratings to numbers
quality_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0, np.nan:0}
for col in ['ExterQual', 'KitchenQual', 'BsmtQual']:
    train[col] = train[col].map(quality_map)
    test[col] = test[col].map(quality_map)

# One-hot encode some categorical features
cat_to_ohe = ['Foundation', 'GarageFinish', 'BsmtFinType1']
train = pd.get_dummies(train, columns=cat_to_ohe, drop_first=True)
test = pd.get_dummies(test, columns=cat_to_ohe, drop_first=True)

# Make sure train and test have the same columns
test = test.reindex(columns=train.columns.drop('SalePrice'), fill_value=0)

X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)
train_pred = model.predict(X)

print("Model Evaluation on training data:")
print(f"MAE: {mean_absolute_error(y, train_pred):.2f}")
rmse = sqrt(mean_squared_error(y, train_pred))
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2_score(y, train_pred):.2f}")

# Folder to save plots
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
os.makedirs(output_dir, exist_ok=True)

# Function to show and save plots
def show_and_save_plot(filename):
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Scatter plot of actual vs predicted prices
plt.figure(figsize=(8,6))
plt.scatter(y, train_pred, alpha=0.5, color='blue')
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
show_and_save_plot("scatter_actual_vs_predicted.png")

# Bar plot of feature importance (coefficients)
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients = coefficients.sort_values()

plt.figure(figsize=(12, 8))
coefficients.plot(kind='barh', color='teal')
plt.title("Feature Importance by Coefficients")
plt.xlabel("Coefficient Value")
show_and_save_plot("feature_importance.png")

# Boxplots for key features
key_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
plt.figure(figsize=(12,6))
train[key_features].boxplot()
plt.title("Boxplot of Key Features")
show_and_save_plot("boxplots.png")

# Correlation heatmap
plt.figure(figsize=(14,12))
corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap")
show_and_save_plot("correlation_heatmap.png")

# Line plot of median SalePrice by year built
median_price_per_year = train.groupby('YearBuilt')['SalePrice'].median()
plt.figure(figsize=(12,6))
median_price_per_year.plot()
plt.title("Median SalePrice by Year Built")
plt.xlabel("Year Built")
plt.ylabel("Median SalePrice")
show_and_save_plot("lineplot_yearbuilt.png")

# Ask user for custom house details and predict its price
def predict_custom_house():
    print("\nEnter details for your house:")
    user_data = {}

    ranges = {
        'OverallQual': "Overall Quality (1-10)",
        'GrLivArea': "Above ground living area in sq ft (500-4000)",
        'GarageCars': "Number of cars garage holds (0-4)",
        'GarageArea': "Garage area in sq ft (0-1500)",
        'TotalBsmtSF': "Total basement area in sq ft (0-3000)",
        '1stFlrSF': "1st Floor area in sq ft (400-2500)",
        'ExterQual': "Exterior Quality (1=Poor to 5=Excellent)",
        'FullBath': "Number of full bathrooms (0-4)",
        'TotRmsAbvGrd': "Total rooms above ground (2-14)",
        'YearBuilt': "Year Built (1872-2010)",
        'KitchenQual': "Kitchen Quality (1=Poor to 5=Excellent)",
        'YearRemodAdd': "Year Remodeled (1950-2010)",
        'BsmtQual': "Basement Quality (1=Poor to 5=Excellent)",
        'MasVnrArea': "Masonry veneer area in sq ft (0-1000)",
        'Fireplaces': "Number of fireplaces (0-3)",
        'GarageYrBlt': "Year garage was built (1900-2010)"
    }

    friendly_map = {
        "Foundation_CBlock": "Foundation type CBlock",
        "Foundation_PConc": "Foundation type PConc",
        "Foundation_Slab": "Foundation type Slab",
        "Foundation_Stone": "Foundation type Stone",
        "Foundation_Wood": "Foundation type Wood",
        "GarageFinish_RFn": "Garage finish Rough Finished (RFn)",
        "GarageFinish_Unf": "Garage finish Unfinished (Unf)",
        "BsmtFinType1_BLQ": "Basement finish type BLQ",
        "BsmtFinType1_GLQ": "Basement finish type GLQ",
        "BsmtFinType1_LwQ": "Basement finish type LwQ",
        "BsmtFinType1_Rec": "Basement finish type Rec",
        "BsmtFinType1_Unf": "Basement finish type Unfinished"
    }

    for feature in X.columns:
        if feature in friendly_map:
            prompt = f"{friendly_map[feature]} (Yes=1, No=0): "
            while True:
                val = input(prompt).strip()
                if val in ['0', '1', '']:
                    user_data[feature] = int(val) if val != '' else 0
                    break
                else:
                    print("Enter 1 for Yes or 0 for No.")
        else:
            prompt = f"{feature} ({ranges.get(feature, 'Number')}): "
            while True:
                val = input(prompt).strip()
                if val == '':
                    val = 0
                try:
                    user_data[feature] = float(val)
                    break
                except ValueError:
                    print("Please enter a valid number.")

    user_df = pd.DataFrame([user_data])
    user_df = user_df.reindex(columns=X.columns, fill_value=0)
    price_pred = model.predict(user_df)[0]
    print(f"\nEstimated House Price: ${price_pred:,.2f}")

if __name__ == "__main__":
    predict_custom_house()
