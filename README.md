# House Price Prediction

This Project was initially made on January 2025 but pushed to github on June 2025
A simple machine learning project to predict house prices using linear regression and stacking models, with interactive user input and visualization of key features.

---

## Project Overview

This project trains regression models on house price data, evaluates their performance, and allows the user to input custom house features to get a predicted price. It also generates and saves various visualizations related to the data and model.

---

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
Setup
Place the dataset files (train.csv and test.csv) in the same directory as main.py.

Run the script:

bash
Copy
Edit
python main.py
What the script does
Loads and preprocesses the dataset.

Handles missing values and encodes categorical features.

Trains multiple models (Ridge, Random Forest, Stacking) and evaluates them.

Displays key evaluation metrics: MAE, RMSE, R².

Creates and saves graphs in a graphs folder:

Actual vs Predicted scatter plot

Feature importance bar plot

Boxplots for key features

Correlation heatmap

Median SalePrice by Year Built line plot

Asks the user to input custom house features and predicts the house price based on the trained model.

Usage
When you run main.py, you will see the training and evaluation outputs. Then, the program will prompt you to enter details about your house (like quality, size, number of rooms, etc.). Input the requested values to get an estimated house price.

Example prompt:

java
Copy
Edit
Enter details for your house:
OverallQual (Overall Quality (1-10)): 7
GrLivArea (Above ground living area in sq ft (500-4000)): 1500
...
Output
Model evaluation results printed in the console.

Plots saved to a graphs directory (created automatically if it doesn't exist).

Estimated house price printed based on user input.

Notes
The quality features are mapped from descriptive ratings (e.g., Excellent = 5, Good = 4) internally.

For categorical features converted to one-hot encoding, you'll be asked simple yes/no (1/0) questions.

Ensure you have the dataset files and dependencies installed before running.
