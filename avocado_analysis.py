import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import time

file_path = "avocado.csv"


<<<<<<< HEAD
def load_data_pandas(file_path):
    """Load the avocado dataset using Pandas."""
    try:
        df = pd.read_csv(file_path, engine="python")
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
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # Adjust the date format
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")  # date format
    # Extract month from the date
    df["month"] = df["Date"].dt.month  # extract month
    return df


def preprocess_data_polars(df):
    """Preprocess the dataset using Polars."""
    df = df.unique()  # Remove duplicates

    # Convert the Date column to a date type
    df = df.with_columns(
        pl.col("Date")
        .str.strptime(pl.Date, "%Y-%m-%d")
        .alias("Date")  # Adjust the date format
    )

    if df["Date"].dtype == pl.Date:
        df = df.with_columns(pl.col("Date").dt.month().alias("month"))  # extract month
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

=======
def load_data(file_path, library='pandas'):
    """Load the avocado dataset using the specified library."""
    print(f"DEBUG: load_data() called with file_path={file_path}, library={library}")
    print(f"Loading data from {file_path} using {library}...")
    try:
        if library == 'pandas':
            return pd.read_csv(file_path, engine="python")
        elif library == 'polars':
            return pl.read_csv(file_path)
        else:
            raise ValueError("Unsupported library. Use 'pandas' or 'polars'.")
    except FileNotFoundError:
        print("ERROR: The specified file was not found!")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def preprocess_data(df, library='pandas'):
    """Preprocess the dataset using the specified library."""
    if library == 'pandas':
        df.drop_duplicates(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        df["month"] = df["Date"].dt.month
    elif library == 'polars':
        df = df.unique()
        df = df.with_columns(
            pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d").alias("Date")
        )
        if df["Date"].dtype == pl.Date:
            df = df.with_columns(pl.col("Date").dt.month().alias("month"))
        else:
            print("ERROR: Date conversion failed. Please check the date format.")
    return df


def filter_chicago_data(df, library='pandas'):
    """Filter the dataset for Chicago avocados using the specified library."""
    if library == 'pandas':
        return df[df["region"] == "Chicago"]
    elif library == 'polars':
        return df.filter(pl.col("region") == "Chicago")

def compute_monthly_average_price(df, library='pandas'):
    """Compute the average avocado price by month using the specified library."""
    if library == 'pandas':
        return df.groupby("month")["AveragePrice"].mean().reset_index()
    elif library == 'polars':
        return df.group_by("month").agg(pl.col("AveragePrice").mean().alias("AveragePrice"))
>>>>>>> c42bf71 (refactoring)

def train_model(X, y):
    """Train a linear regression model and return the trained model."""
    model = LinearRegression()
<<<<<<< HEAD
    # Train the model; X is the feature (month), y is the target (AveragePrice)
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance using MAE and R²."""
    # Predict the target values for the test set
    y_pred = model.predict(X_test)
    # Mean Absolute Error: average magnitude of the errors
    mae = mean_absolute_error(y_test, y_pred)
    # R-squared: proportion of variance explained by the model
    r2 = r2_score(y_test, y_pred)
    return mae, r2


def plot_results(X_test, y_test, y_pred, library):
    """Plot the actual vs predicted prices."""
    # Set the figure size for better visibility
    plt.figure(figsize=(10, 6))
    # Scatter plot for actual prices
    plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
    # Scatter plot for predicted prices
    plt.scatter(X_test, y_pred, color="red", label="Predicted Prices")
    # Regression line
=======
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance using MAE and R²."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2

def plot_results(X_test, y_test, y_pred, library):
    """Plot the actual vs predicted prices."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
    plt.scatter(X_test, y_pred, color="red", label="Predicted Prices")
>>>>>>> c42bf71 (refactoring)
    plt.plot(X_test, y_pred, color="green", linewidth=2, label="Regression Line")
    plt.title(f"Avocado Prices in Chicago Based on Month ({library})")
    plt.xlabel("Month")
    plt.ylabel("Average Price")
    plt.legend()
    plt.show()

<<<<<<< HEAD

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
=======
def run_analysis(file_path, library):
    """Run the analysis using the specified library (Pandas or Polars)."""
    df = load_data(file_path, library)
    if df is not None:
        df = preprocess_data(df, library)
        chicago_avocados = filter_chicago_data(df, library)
        monthly_avg_price = compute_monthly_average_price(chicago_avocados, library)

        # Prepare data for modeling
        if library == 'pandas':
            X = monthly_avg_price[["month"]]
            y = monthly_avg_price["AveragePrice"]
        else:  # Polars
            X = monthly_avg_price.select("month").to_numpy().reshape(-1, 1)
            y = monthly_avg_price.select("AveragePrice").to_numpy().ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>>>>>> c42bf71 (refactoring)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        mae, r2 = evaluate_model(model, X_test, y_test)
<<<<<<< HEAD
        print(f"Polars - Mean Absolute Error: {mae:.2f}")
        print(f"Polars - R-squared: {r2:.2f}")

        # Plot results
        plot_results(X_test, y_test, model.predict(X_test), "Polars")

=======
        print(f"{library.capitalize()} - Mean Absolute Error: {mae:.2f}")
        print(f"{library.capitalize()} - R-squared: {r2:.2f}")

        # Plot results
        plot_results(X_test, y_test, model.predict(X_test), library)
>>>>>>> c42bf71 (refactoring)

if __name__ == "__main__":
    file_path = "avocado.csv"

    # Run analysis with Pandas
    print("Running analysis with Pandas...")
    start_time = time.time()
<<<<<<< HEAD
    run_analysis_with_pandas(file_path)
=======
    run_analysis(file_path, 'pandas')
>>>>>>> c42bf71 (refactoring)
    pandas_duration = time.time() - start_time
    print(f"Pandas analysis completed in {pandas_duration:.2f} seconds.\n")

    # Run analysis with Polars
    print("Running analysis with Polars...")
    start_time = time.time()
<<<<<<< HEAD
    run_analysis_with_polars(file_path)
    polars_duration = time.time() - start_time
    print(f"Polars analysis completed in {polars_duration:.2f} seconds.")
=======
    run_analysis(file_path, 'polars')
    polars_duration = time.time() - start_time
    print(f"Polars analysis completed in {polars_duration:.2f} seconds.")
>>>>>>> c42bf71 (refactoring)
