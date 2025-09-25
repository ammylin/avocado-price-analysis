import unittest
import pandas as pd
import polars as pl
<<<<<<< HEAD
from avocado_analysis import (
    load_data_pandas,
    load_data_polars,
    preprocess_data_pandas,
    preprocess_data_polars,
    filter_chicago_data_pandas,
    filter_chicago_data_polars,
    compute_monthly_average_price_pandas,
    compute_monthly_average_price_polars,
=======

from avocado_analysis import (
    load_data,
    preprocess_data,
    filter_chicago_data,
    compute_monthly_average_price,
>>>>>>> c42bf71 (refactoring)
    train_model,
    evaluate_model,
)

<<<<<<< HEAD

class TestAvocadoModel(
    unittest.TestCase
):  # Test case for the avocado analysis using Pandas.
=======
class TestAvocadoModel(unittest.TestCase):
    """Test case for the avocado analysis using Pandas."""
>>>>>>> c42bf71 (refactoring)

    def setUp(self):
        # Sample data for testing
        data = {
            "Date": [
                "2021-01-01",
                "2021-01-08",
                "2021-01-15",
                "2021-01-22",
                "2021-01-29",
            ],
            "AveragePrice": [1.00, 1.10, 1.20, 1.30, 1.40],
            "region": ["Chicago", "Chicago", "Chicago", "Chicago", "Chicago"],
        }
        self.df = pd.DataFrame(data)

    def test_load_data(self):
        # Test loading data from a valid file
<<<<<<< HEAD
        # Ensure this file exists for the test
        df = load_data_pandas("avocado.csv")
        # Check that the DataFrame is not None
        self.assertIsNotNone(df)
        # Ensure that the DataFrame has rows
        self.assertGreater(df.shape[0], 0)
        # Check that the DataFrame has the expected number of columns
        self.assertEqual(df.shape[1], 14)

    def test_preprocess_data(self):
        #  Test preprocessing data
        # Use copy to avoid modifying the original data
        df_processed = preprocess_data_pandas(self.df.copy())
        # No duplicates
        self.assertEqual(df_processed.shape[0], 5)
        # Check the date format
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_processed["Date"]))

    def test_filter_chicago_data(self):
        #  Test filtering for Chicago data
        # Use copy to avoid modifying the original data
        df_filtered = filter_chicago_data_pandas(self.df.copy())
        # Check if the number of rows is as expected for Chicago data
        self.assertEqual(df_filtered.shape[0], 5)

    def test_compute_monthly_average_price(self):
        #  Test computing monthly average price
        # Ensure preprocessing
        df_processed = preprocess_data_pandas(self.df.copy())
        df_filtered = filter_chicago_data_pandas(df_processed)
        monthly_avg = compute_monthly_average_price_pandas(df_filtered)
        # Check if there is only one month in the sample data
        self.assertEqual(monthly_avg.shape[0], 1)
        # Check the average price
        self.assertAlmostEqual(monthly_avg["AveragePrice"].iloc[0], 1.20)

    def test_train_model(self):
        # Test training the model
        # Ensure preprocessing
        df_processed = preprocess_data_pandas(self.df.copy())
        df_filtered = filter_chicago_data_pandas(df_processed)
        monthly_avg = compute_monthly_average_price_pandas(df_filtered)
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        # Check that the model is not None
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        # Test model evaluation
        # Ensure preprocessing
        df_processed = preprocess_data_pandas(self.df.copy())
        df_filtered = filter_chicago_data_pandas(df_processed)
        monthly_avg = compute_monthly_average_price_pandas(df_filtered)
=======
        df = load_data("avocado.csv", library='pandas')
        self.assertIsNotNone(df, "DataFrame should not be None")
        self.assertGreater(df.shape[0], 0, "DataFrame should have rows")
        self.assertEqual(df.shape[1], 14, "DataFrame should have 14 columns")

    def test_preprocess_data(self):
        # Test preprocessing data
        df_processed = preprocess_data(self.df.copy(), library='pandas')
        self.assertEqual(df_processed.shape[0], 5, "No duplicates should exist")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_processed["Date"]), "Date column should be in datetime format")

    def test_filter_chicago_data(self):
        # Test filtering for Chicago data
        df_filtered = filter_chicago_data(self.df.copy(), library='pandas')
        self.assertEqual(df_filtered.shape[0], 5, "Filtered data should have 5 rows for Chicago")

    def test_compute_monthly_average_price(self):
        # Test computing monthly average price
        df_processed = preprocess_data(self.df.copy(), library='pandas')
        df_filtered = filter_chicago_data(df_processed, library='pandas')
        monthly_avg = compute_monthly_average_price(df_filtered, library='pandas')
        self.assertEqual(monthly_avg.shape[0], 1, "There should be one month in the sample data")
        self.assertAlmostEqual(monthly_avg["AveragePrice"].iloc[0], 1.20, "Average price should be approximately 1.20")

    def test_train_model(self):
        # Test training the model
        df_processed = preprocess_data(self.df.copy(), library='pandas')
        df_filtered = filter_chicago_data(df_processed, library='pandas')
        monthly_avg = compute_monthly_average_price(df_filtered, library='pandas')
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        self.assertIsNotNone(model, "Model should not be None")

    def test_evaluate_model(self):
        # Test model evaluation
        df_processed = preprocess_data(self.df.copy(), library='pandas')
        df_filtered = filter_chicago_data(df_processed, library='pandas')
        monthly_avg = compute_monthly_average_price(df_filtered, library='pandas')
>>>>>>> c42bf71 (refactoring)
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        mae, r2 = evaluate_model(model, X, y)
<<<<<<< HEAD
        # Check that MAE is a float
        self.assertIsInstance(mae, float)
        # Check that R² is a float
        self.assertIsInstance(r2, float)


class TestAvocadoModelPolars(unittest.TestCase):

    def setUp(self):
        #  Sample data for testing
=======
        self.assertIsInstance(mae, float, "MAE should be a float")
        self.assertIsInstance(r2, float, "R² should be a float")


class TestAvocadoModelPolars(unittest.TestCase):
    """Test case for the avocado analysis using Polars."""

    def setUp(self):
        # Sample data for testing
>>>>>>> c42bf71 (refactoring)
        data = {
            "Date": [
                "2021-01-01",
                "2021-01-08",
                "2021-01-15",
                "2021-01-22",
                "2021-01-29",
            ],
            "AveragePrice": [1.00, 1.10, 1.20, 1.30, 1.40],
            "region": ["Chicago", "Chicago", "Chicago", "Chicago", "Chicago"],
        }
        self.df = pl.DataFrame(data)

    def test_load_data(self):
        # Test loading data from a valid file
<<<<<<< HEAD
        # Ensure this file exists for the test
        df = load_data_polars("avocado.csv")
        self.assertIsNotNone(df)  # Check that the DataFrame is not None
        self.assertGreater(df.height, 0)  # Ensure that the DataFrame has rows
        self.assertEqual(
            df.width, 14
        )  # Check that the DataFrame has the expected number of columns

    def test_preprocess_data(self):
        # Test preprocessing data
        df_processed = preprocess_data_polars(self.df.clone())
        # Check that there are no duplicates
        self.assertEqual(df_processed.height, 5)
        # Check the date format
        self.assertTrue(df_processed["Date"].dtype == pl.Date)

    def test_filter_chicago_data(self):
        # Test filtering for Chicago data
        df_filtered = filter_chicago_data_polars(self.df.clone())
        self.assertEqual(
            df_filtered.height, 5
        )  # Check if the number of rows is as expected for Chicago data

    def test_compute_monthly_average_price(self):
        # Test computing monthly average price
        # Ensure preprocessing
        df_processed = preprocess_data_polars(self.df.clone())
        df_filtered = filter_chicago_data_polars(df_processed)
        monthly_avg = compute_monthly_average_price_polars(df_filtered)
        self.assertEqual(
            monthly_avg.height, 1
        )  # Check if there is only one month in the sample data
        self.assertAlmostEqual(
            monthly_avg["AveragePrice"][0], 1.20
        )  # Check the average price

    def test_train_model(self):
        # Test training the model
        df_processed = preprocess_data_polars(self.df.clone())  # Ensure preprocessing
        df_filtered = filter_chicago_data_polars(df_processed)
        monthly_avg = compute_monthly_average_price_polars(df_filtered)
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        self.assertIsNotNone(model)  # Check that the model is not None

    def test_evaluate_model(self):
        # Test model evaluation
        df_processed = preprocess_data_polars(self.df.clone())  # Ensure preprocessing
        df_filtered = filter_chicago_data_polars(df_processed)
        monthly_avg = compute_monthly_average_price_polars(df_filtered)
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        mae, r2 = evaluate_model(model, X, y)
        self.assertIsInstance(mae, float)  # Check that MAE is a float
        self.assertIsInstance(r2, float)  # Check that R² is a float


if __name__ == "__main__":
    unittest.main()  # Run the unit tests
=======
        df = load_data("avocado.csv", library='polars')
        self.assertIsNotNone(df, "DataFrame should not be None")
        self.assertGreater(df.height, 0, "DataFrame should have rows")
        self.assertEqual(df.width, 14, "DataFrame should have 14 columns")

    def test_preprocess_data(self):
        # Test preprocessing data
        df_processed = preprocess_data(self.df.clone(), library='polars')
        self.assertEqual(df_processed.height, 5, "No duplicates should exist")
        self.assertTrue(df_processed["Date"].dtype == pl.Date, "Date column should be in Date format")

    def test_filter_chicago_data(self):
        # Test filtering for Chicago data
        df_filtered = filter_chicago_data(self.df.clone(), library='polars')
        self.assertEqual(df_filtered.height, 5, "Filtered data should have 5 rows for Chicago")

    def test_compute_monthly_average_price(self):
        # Test computing monthly average price
        df_processed = preprocess_data(self.df.clone(), library='polars')
        df_filtered = filter_chicago_data(df_processed, library='polars')
        monthly_avg = compute_monthly_average_price(df_filtered, library='polars')
        self.assertEqual(monthly_avg.height, 1, "There should be one month in the sample data")
        self.assertAlmostEqual(monthly_avg["AveragePrice"][0], 1.20, "Average price should be approximately 1.20")

    def test_train_model(self):
        # Test training the model
        df_processed = preprocess_data(self.df.clone(), library='polars')
        df_filtered = filter_chicago_data(df_processed, library='polars')
        monthly_avg = compute_monthly_average_price(df_filtered, library='polars')
        X = monthly_avg.select("month").to_numpy().reshape(-1, 1)
        y = monthly_avg.select("AveragePrice").to_numpy().ravel()
        model = train_model(X, y)
        self.assertIsNotNone(model, "Model should not be None")

    def test_evaluate_model(self):
        # Test model evaluation
        df_processed = preprocess_data(self.df.clone(), library='polars')
        df_filtered = filter_chicago_data(df_processed, library='polars')
        monthly_avg = compute_monthly_average_price(df_filtered, library='polars')
        X = monthly_avg.select("month").to_numpy().reshape(-1, 1)
        y = monthly_avg.select("AveragePrice").to_numpy().ravel()
        model = train_model(X, y)
        mae, r2 = evaluate_model(model, X, y)
        self.assertIsInstance(mae, float, "MAE should be a float")
        self.assertIsInstance(r2, float, "R² should be a float")


if __name__ == "__main__":
    unittest.main()  # Run the unit tests
>>>>>>> c42bf71 (refactoring)
