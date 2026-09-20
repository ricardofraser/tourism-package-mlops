
# Import pandas for data handling
import pandas as pd

# Import train_test_split for creating training and testing datasets
from sklearn.model_selection import train_test_split

from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

# Define the path to the registered raw tourism dataset
RAW_PATH = DATA_DIR / "tourism.csv"


# Load the registered raw dataset
df = pd.read_csv(RAW_PATH)


# Remove exported dataframe index
# Unnamed: 0 is an exported dataframe index
df.drop(columns=['Unnamed: 0'], inplace=True)


# Correct the inconsistent gender label identified during exploratory data analysis
# "Fe Male" represents the same category as "Female"
df["Gender"] = df["Gender"].replace({
    "Fe Male": "Female"
})


# Distribution of target before duplicate removal
target_distribution = pd.DataFrame({
    "Count": df["ProdTaken"].value_counts().sort_index(),
    "Percentage": (
        df["ProdTaken"]
        .value_counts(normalize=True)
        .sort_index() * 100
    ).round(2)
})

print("\n Target Distribution before duplicate removal")
print(target_distribution)
print("\n -------------")


# Remove exact duplicate records identified during exploratory data analysis
# This prevents repeated observations from appearing in the modelling dataset
# and potentially inflating model performance
rows_before = len(df)

df = df.drop_duplicates().reset_index(drop=True)

rows_after = len(df)
duplicates_removed = rows_before - rows_after


# Remove non-predictive customer identifier
# CustomerID uniquely identifies a customer but does not describe customer behaviour
df.drop(columns=['CustomerID'], inplace=True)


# Define the target variable
# ProdTaken = 1 means the customer purchased the tourism package
# ProdTaken = 0 means the customer did not purchase the package
target = "ProdTaken"


# Separate the predictor variables from the target variable
X = df.drop(columns=[target])
y = df[target]


# Split the dataset into training and testing sets
# test_size=0.20 reserves 20% of the data for final testing
# random_state=42 makes the split reproducible
# stratify=y preserves the class distribution of ProdTaken across both datasets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Save the prepared datasets locally
# These files will later be used by the model-training stage
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)


# Confirm that the preparation step completed successfully
print("Data prepared successfully.")

# Display the cleaning results
print(f"Rows before duplicate removal: {rows_before}")
print(f"Rows after duplicate removal:  {rows_after}")
print(f"Duplicate rows removed:        {duplicates_removed}")

# Confirm that no exact duplicates remain
print(f"Remaining duplicate rows:      {df.duplicated().sum()}")

# Confirm that the inconsistent gender category was corrected
print("\nGender categories after cleaning:")
print(df["Gender"].value_counts())


# Display the resulting dataset dimensions
print(f"\nXtrain shape: {Xtrain.shape}")
print(f"Xtest shape:  {Xtest.shape}")
print(f"ytrain shape: {ytrain.shape}")
print(f"ytest shape:  {ytest.shape}")


# Display the class distribution in the training target
print("\nProdTaken distribution in training data:")

train_distribution = pd.DataFrame({
    "Count": ytrain.value_counts().sort_index(),
    "Percentage": (
        ytrain.value_counts(normalize=True).sort_index() * 100
    ).round(2)
})

print(train_distribution)


# Display the class distribution in the testing target
print("\nProdTaken distribution in testing data:")

test_distribution = pd.DataFrame({
    "Count": ytest.value_counts().sort_index(),
    "Percentage": (
        ytest.value_counts(normalize=True).sort_index() * 100
    ).round(2)
})

print(test_distribution)


Xtrain.to_csv(DATA_DIR / "Xtrain.csv", index=False)
Xtest.to_csv(DATA_DIR / "Xtest.csv", index=False)
ytrain.rename("ProdTaken").to_csv(DATA_DIR / "ytrain.csv", index=False)
ytest.rename("ProdTaken").to_csv(DATA_DIR / "ytest.csv", index=False)
