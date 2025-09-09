import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import time

file_path = "avocado.csv"


def load_data_pandas(file_path):
    """Load the avocado dataset using Pandas."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("ERROR: The specified file was not found!")
        return None


def load_data_polars(file_path):
    """Load the avocado dataset using Polars."""
    try:
        df = pl.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("ERROR: The specified file was not found!")
        return None


def preprocess_data_pandas(df):
    """Preprocess the dataset using Pandas."""
    df.drop_duplicates(inplace=True) # Remove duplicates
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d") # Adjust the date format
    df["month"] = df["Date"].dt.month # Extract month from the date
    return df


def preprocess_data_polars(df):
    """Preprocess the dataset using Polars."""
    df = df.unique() # Remove duplicates


    # Convert the Date column to a date type
    df = df.with_columns(
        pl.col("Date")
        .str.strptime(pl.Date, "%Y-%m-%d")
        .alias("Date")  # Adjust the date format
    )

    if df["Date"].dtype == pl.Date:
        df = df.with_columns(
            pl.col("Date").dt.month().alias("month")  # Extract month from the date
        )
    else:
        print("ERROR: Date conversion failed. Please check the date format.")
    return df


def filter_chicago_data_pandas(df):
    """Filter the dataset for Chicago avocados using Pandas."""
    return df[df["region"] == "Chicago"]


def filter_chicago_data_polars(df):
    """Filter the dataset for Chicago avocados using Polars."""
    return df.filter(pl.col("region") == "Chicago")


def compute_monthly_average_price_pandas(df):
    """Compute the average avocado price by month using Pandas."""
    return df.groupby("month")["AveragePrice"].mean().reset_index()


def compute_monthly_average_price_polars(df):
    """Compute the average avocado price by month using Polars."""
    return df.group_by("month").agg(pl.col("AveragePrice").mean().alias("AveragePrice"))


def train_model(X, y):
    """Train a linear regression model and return the trained model."""
    model = LinearRegression()
    model.fit(X, y) # Train the model; X is the feature (month), y is the target (AveragePrice)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance using MAE and R²."""
    y_pred = model.predict(X_test) # Predict the target values for the test set
    mae = mean_absolute_error(y_test, y_pred) # Mean Absolute Error, which measures the average magnitude of the errors in a set of predictions, without considering their direction.
    r2 = r2_score(y_test, y_pred) # R-squared, which indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
    return mae, r2


def plot_results(X_test, y_test, y_pred, library):
    """Plot the actual vs predicted prices."""
    plt.figure(figsize=(10, 6)) # Set the figure size for better visibility
    plt.scatter(X_test, y_test, color="blue", label="Actual Prices")   # Scatter plot for actual prices
    plt.scatter(X_test, y_pred, color="red", label="Predicted Prices")  # Scatter plot for predicted prices
    plt.plot(X_test, y_pred, color="green", linewidth=2, label="Regression Line")   # Regression line
    plt.title(f"Avocado Prices in Chicago Based on Month ({library})") 
    plt.xlabel("Month")         
    plt.ylabel("Average Price")
    plt.legend()
    plt.show()


def run_analysis_with_pandas(file_path):
    """Run the analysis using Pandas."""
    df = load_data_pandas(file_path)   
    if df is not None:
        df = preprocess_data_pandas(df)
        chicago_avocados = filter_chicago_data_pandas(df)
        monthly_avg_price = compute_monthly_average_price_pandas(chicago_avocados)

        # Prepare data for modeling
        X = monthly_avg_price[["month"]]
        y = monthly_avg_price["AveragePrice"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        mae, r2 = evaluate_model(model, X_test, y_test) 
        print(f"Pandas - Mean Absolute Error: {mae:.2f}")
        print(f"Pandas - R-squared: {r2:.2f}")

        # Plot results
        plot_results(X_test, y_test, model.predict(X_test), "Pandas")


def run_analysis_with_polars(file_path):
    """Run the analysis using Polars."""
    df = load_data_polars(file_path)
    if df is not None:
        df = preprocess_data_polars(df)
        chicago_avocados = filter_chicago_data_polars(df)
        monthly_avg_price = compute_monthly_average_price_polars(chicago_avocados)

        # Prepare data for modeling
        X = monthly_avg_price.select("month").to_numpy().reshape(-1, 1)
        y = monthly_avg_price.select("AveragePrice").to_numpy().ravel()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        mae, r2 = evaluate_model(model, X_test, y_test)
        print(f"Polars - Mean Absolute Error: {mae:.2f}")
        print(f"Polars - R-squared: {r2:.2f}")

        # Plot results
        plot_results(X_test, y_test, model.predict(X_test), "Polars")


if __name__ == "__main__":
    file_path = "avocado.csv"

    # Run analysis with Pandas
    print("Running analysis with Pandas...")
    start_time = time.time()
    run_analysis_with_pandas(file_path)
    pandas_duration = time.time() - start_time
    print(f"Pandas analysis completed in {pandas_duration:.2f} seconds.\n")

    # Run analysis with Polars
    print("Running analysis with Polars...")
    start_time = time.time()
    run_analysis_with_polars(file_path)
    polars_duration = time.time() - start_time
    print(f"Polars analysis completed in {polars_duration:.2f} seconds.")
