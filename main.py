import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import os

# === Load your data ===
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# === Select important features ===
top_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
    '1stFlrSF', 'ExterQual', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
    'KitchenQual', 'GarageFinish', 'YearRemodAdd', 'BsmtQual', 'Foundation',
    'MasVnrArea', 'Fireplaces', 'GarageYrBlt', 'BsmtFinType1'
]

train = train[top_features + ['SalePrice']]
test = test[top_features]

# === Fill missing values ===
num_cols = train.select_dtypes(include=[np.number]).columns.drop('SalePrice')
cat_cols = train.select_dtypes(include=['object']).columns

for col in num_cols:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)

for col in cat_cols:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col] = test[col].fillna(mode_val)

# === Map quality ratings ===
quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, np.nan: 0}
for col in ['ExterQual', 'KitchenQual', 'BsmtQual']:
    train[col] = train[col].map(quality_map)
    test[col] = test[col].map(quality_map)

# === One-hot encode categorical ===
cat_to_ohe = ['Foundation', 'GarageFinish', 'BsmtFinType1']
train = pd.get_dummies(train, columns=cat_to_ohe, drop_first=True)
test = pd.get_dummies(test, columns=cat_to_ohe, drop_first=True)

# === Align test columns with train ===
test = test.reindex(columns=train.columns.drop('SalePrice'), fill_value=0)

X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# === Train linear regression ===
model = LinearRegression()
model.fit(X, y)
train_pred = model.predict(X)

print("\n=== Model Evaluation ===")
print(f"MAE: {mean_absolute_error(y, train_pred):.2f}")
print(f"RMSE: {sqrt(mean_squared_error(y, train_pred)):.2f}")
print(f"R2: {r2_score(y, train_pred):.2f}")

# === Create 'graphs' folder ===
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
os.makedirs(output_dir, exist_ok=True)

# === Safe plot save and show ===
def show_and_save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))  # ✅ save first!
    plt.show()
    plt.close()

# === 1. Scatter plot ===
plt.figure(figsize=(8,6))
plt.scatter(y, train_pred, alpha=0.5, color='blue')
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
show_and_save_plot("scatter_actual_vs_predicted.png")

# === 2. Feature importance ===
coefficients = pd.Series(model.coef_, index=X.columns).sort_values()
plt.figure(figsize=(12, 8))
coefficients.plot(kind='barh', color='teal')
plt.title("Feature Importance (Coefficients)")
plt.xlabel("Coefficient Value")
show_and_save_plot("feature_importance.png")

# === 3. Boxplots ===
key_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
plt.figure(figsize=(12,6))
train[key_features].boxplot()
plt.title("Boxplot of Key Features")
show_and_save_plot("boxplots.png")

# === 4. Correlation Heatmap ===
plt.figure(figsize=(14,12))
corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap")
show_and_save_plot("correlation_heatmap.png")

# === 5. Median SalePrice by Year Built ===
median_price_per_year = train.groupby('YearBuilt')['SalePrice'].median()
plt.figure(figsize=(12,6))
median_price_per_year.plot()
plt.title("Median SalePrice by Year Built")
plt.xlabel("Year Built")
plt.ylabel("Median SalePrice")
show_and_save_plot("lineplot_yearbuilt.png")

# === User input to predict ===
def predict_custom_house():
    print("\nEnter your house details:")
    user_data = {}

    ranges = {
        'OverallQual': "Overall Quality (1-10)",
        'GrLivArea': "Above ground living area sq ft (500-4000)",
        'GarageCars': "Garage car capacity (0-4)",
        'GarageArea': "Garage area sq ft (0-1500)",
        'TotalBsmtSF': "Basement area sq ft (0-3000)",
        '1stFlrSF': "1st Floor area sq ft (400-2500)",
        'ExterQual': "Exterior Quality (1=Poor to 5=Excellent)",
        'FullBath': "Number of full bathrooms (0-4)",
        'TotRmsAbvGrd': "Total rooms above ground (2-14)",
        'YearBuilt': "Year Built (1872-2010)",
        'KitchenQual': "Kitchen Quality (1=Poor to 5=Excellent)",
        'YearRemodAdd': "Year Remodeled (1950-2010)",
        'BsmtQual': "Basement Quality (1=Poor to 5=Excellent)",
        'MasVnrArea': "Masonry veneer area sq ft (0-1000)",
        'Fireplaces': "Number of fireplaces (0-3)",
        'GarageYrBlt': "Year garage was built (1900-2010)"
    }

    friendly_map = {
        "Foundation_CBlock": "Foundation type CBlock (Yes=1, No=0)",
        "Foundation_PConc": "Foundation type PConc (Yes=1, No=0)",
        "Foundation_Slab": "Foundation type Slab (Yes=1, No=0)",
        "Foundation_Stone": "Foundation type Stone (Yes=1, No=0)",
        "Foundation_Wood": "Foundation type Wood (Yes=1, No=0)",
        "GarageFinish_RFn": "Garage finish Rough Finished (Yes=1, No=0)",
        "GarageFinish_Unf": "Garage finish Unfinished (Yes=1, No=0)",
        "BsmtFinType1_BLQ": "Basement finish type BLQ (Yes=1, No=0)",
        "BsmtFinType1_GLQ": "Basement finish type GLQ (Yes=1, No=0)",
        "BsmtFinType1_LwQ": "Basement finish type LwQ (Yes=1, No=0)",
        "BsmtFinType1_Rec": "Basement finish type Rec (Yes=1, No=0)",
        "BsmtFinType1_Unf": "Basement finish type Unfinished (Yes=1, No=0)"
    }

    for feature in X.columns:
        if feature in friendly_map:
            prompt = f"{friendly_map[feature]}: "
            while True:
                val = input(prompt).strip()
                if val in ['0', '1', '']:
                    user_data[feature] = int(val) if val != '' else 0
                    break
                else:
                    print("Enter 1 or 0.")
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
                    print("Enter a valid number.")

    user_df = pd.DataFrame([user_data])
    user_df = user_df.reindex(columns=X.columns, fill_value=0)
    price_pred = model.predict(user_df)[0]
    print(f"\nEstimated House Price: ${price_pred:,.2f}")

if __name__ == "__main__":
    predict_custom_house()
