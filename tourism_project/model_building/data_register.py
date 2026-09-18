# Import pandas to load and validate the raw dataset
import pandas as pd

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
# Define the path to the raw tourism dataset
RAW_PATH = DATA_DIR / "tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Define the complete set of columns expected in the tourism dataset
expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

# Check whether any expected columns are missing from the dataset
missing_columns = [
    column for column in expected_columns
    if column not in df.columns
]

# Stop execution if required columns are missing
if missing_columns:
    raise ValueError(
        f"Dataset is missing expected columns: {missing_columns}"
    )

# Check whether the dataset contains any unexpected columns
unexpected_columns = [
    column for column in df.columns
    if column not in expected_columns
]

# Report unexpected columns without stopping execution
if unexpected_columns:
    print(
        f"Warning: Dataset contains unexpected columns: "
        f"{unexpected_columns}"
    )

# Confirm successful dataset registration
print("Dataset registered successfully.")

# Report the shape of the dataset
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Display the registered column names
print("\nColumns:")
print(df.columns.tolist())

# Display the class distribution of the target variable
print("\nProdTaken distribution:")
print(df["ProdTaken"].value_counts())
