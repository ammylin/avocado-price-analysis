import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import time

file_path = "data/avocado.csv"


def load_data(file_path, library="pandas"):
    """Load the avocado dataset using the specified library."""
    try:
        if library == "pandas":
            return pd.read_csv(file_path, engine="python")
        elif library == "polars":
            return pl.read_csv(file_path)
        else:
            raise ValueError("Unsupported library. Use 'pandas' or 'polars'.")
    except FileNotFoundError:
        print("ERROR: The specified file was not found!")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def preprocess_data(df, library="pandas"):
    """Preprocess the dataset using the specified library."""
    if library == "pandas":
        df.drop_duplicates(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        df["month"] = df["Date"].dt.month
    elif library == "polars":
        df = df.unique()
        try:
            df = df.with_columns(
                pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d").alias("Date")
            )
            if df["Date"].dtype == pl.Date:
                df = df.with_columns(pl.col("Date").dt.month().alias("month"))
            else:
                print("ERROR: Date conversion failed. Please check the date format.")
        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
    return df


def filter_chicago_data(df, library="pandas"):
    """Filter the dataset for Chicago avocados using the specified library."""
    if library == "pandas":
        return df[df["region"] == "Chicago"]
    elif library == "polars":
        return df.filter(pl.col("region") == "Chicago")


def compute_monthly_average_price(df, library="pandas"):
    """Compute the average avocado price by month using the specified library."""
    if library == "pandas":
        return df.groupby("month")["AveragePrice"].mean().reset_index()
    elif library == "polars":
        return df.group_by("month").agg(
            pl.col("AveragePrice").mean().alias("AveragePrice")
        )


def train_model(X, y):
    """Train a linear regression model and return the trained model."""
    model = LinearRegression()
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
    plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Prices")
    plt.title(f"Avocado Prices in Chicago Based on Month ({library})")
    plt.xlabel("Month")
    plt.ylabel("Average Price")
    plt.legend()
    plt.show()


def run_analysis(file_path, library):
    """Run the analysis using the specified library (Pandas or Polars)."""
    df = load_data(file_path, library)
    if df is not None:
        df = preprocess_data(df, library)
        chicago_avocados = filter_chicago_data(df, library)
        monthly_avg_price = compute_monthly_average_price(chicago_avocados, library)

        # Prepare data for modeling
        if library == "pandas":
            X = monthly_avg_price[["month"]]
            y = monthly_avg_price["AveragePrice"]
        else:  # Polars
            X = monthly_avg_price.select("month").to_numpy().reshape(-1, 1)
            y = monthly_avg_price.select("AveragePrice").to_numpy().ravel()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        mae, r2 = evaluate_model(model, X_test, y_test)
        print(f"{library.capitalize()} - Mean Absolute Error: {mae:.2f}")
        print(f"{library.capitalize()} - R-squared: {r2:.2f}")

        # Plot results
        plot_results(X_test, y_test, model.predict(X_test), library)


if __name__ == "__main__":
    # Run analysis with Pandas
    print("Running analysis with Pandas...")
    start_time = time.time()
    run_analysis(file_path, "pandas")
    pandas_duration = time.time() - start_time
    print(f"Pandas analysis completed in {pandas_duration:.2f} seconds.\n")

    # Run analysis with Polars
    print("Running analysis with Polars...")
    start_time = time.time()
    run_analysis(file_path, "polars")
    polars_duration = time.time() - start_time
    print(f"Polars analysis completed in {polars_duration:.2f} seconds.")
