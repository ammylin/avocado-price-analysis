import unittest
import pandas as pd
import polars as pl
from avocado_analysis import (
    load_data_pandas,
    load_data_polars,
    preprocess_data_pandas,
    preprocess_data_polars,
    filter_chicago_data_pandas,
    filter_chicago_data_polars,
    compute_monthly_average_price_pandas,
    compute_monthly_average_price_polars,
    train_model,
    evaluate_model
)


class TestAvocadoModel(unittest.TestCase):

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
        df = load_data_pandas("avocado.csv")  # Ensure this file exists for the test
        self.assertIsNotNone(df)  # Check that the DataFrame is not None
        self.assertGreater(df.shape[0], 0)  # Ensure that the DataFrame has rows
        self.assertEqual(
            df.shape[1], 14
        )  # Check that the DataFrame has the expected number of columns

    def test_preprocess_data(self):
        # Test preprocessing data
        df_processed = preprocess_data_pandas(self.df.copy())
        self.assertEqual(df_processed.shape[0], 5)  # No duplicates
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_processed["Date"]))

    def test_filter_chicago_data(self):
        # Test filtering for Chicago data
        df_filtered = filter_chicago_data_pandas(self.df.copy())
        self.assertEqual(df_filtered.shape[0], 5)  # All rows are from Chicago

    def test_compute_monthly_average_price(self):
        # Test computing monthly average price
        df_processed = preprocess_data_pandas(self.df.copy())  # Ensure preprocessing
        df_filtered = filter_chicago_data_pandas(df_processed)
        monthly_avg = compute_monthly_average_price_pandas(df_filtered)
        self.assertEqual(monthly_avg.shape[0], 1)  # Only one month in the sample data
        self.assertAlmostEqual(
            monthly_avg["AveragePrice"].iloc[0], 1.20
        )  # Average price

    def test_train_model(self):
        # Test training the model
        df_processed = preprocess_data_pandas(self.df.copy())  # Ensure preprocessing
        df_filtered = filter_chicago_data_pandas(df_processed)
        monthly_avg = compute_monthly_average_price_pandas(df_filtered)
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        # Test model evaluation
        df_processed = preprocess_data_pandas(self.df.copy())  # Ensure preprocessing
        df_filtered = filter_chicago_data_pandas(df_processed)
        monthly_avg = compute_monthly_average_price_pandas(df_filtered)
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        mae, r2 = evaluate_model(model, X, y)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(r2, float)


class TestAvocadoModelPolars(unittest.TestCase):

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
        self.df = pl.DataFrame(data)

    def test_load_data(self):
        # Test loading data from a valid file
        df = load_data_polars("avocado.csv")  # Ensure this file exists for the test
        self.assertIsNotNone(df)  # Check that the DataFrame is not None
        self.assertGreater(df.height, 0)  # Ensure that the DataFrame has rows
        self.assertEqual(
            df.width, 14
        )  # Check that the DataFrame has the expected number of columns

    def test_preprocess_data(self):
        # Test preprocessing data
        df_processed = preprocess_data_polars(self.df.clone())
        self.assertEqual(df_processed.height, 5)  # No duplicates
        self.assertTrue(df_processed["Date"].dtype == pl.Date)

    def test_filter_chicago_data(self):
        # Test filtering for Chicago data
        df_filtered = filter_chicago_data_polars(self.df.clone())
        self.assertEqual(df_filtered.height, 5)  # All rows are from Chicago

    def test_compute_monthly_average_price(self):
        # Test computing monthly average price
        df_processed = preprocess_data_polars(self.df.clone())  # Ensure preprocessing
        df_filtered = filter_chicago_data_polars(df_processed)
        monthly_avg = compute_monthly_average_price_polars(df_filtered)
        self.assertEqual(monthly_avg.height, 1)  # Only one month in the sample data
        self.assertAlmostEqual(monthly_avg["AveragePrice"][0], 1.20)  # Average price

    def test_train_model(self):
        # Test training the model
        df_processed = preprocess_data_polars(self.df.clone())  # Ensure preprocessing
        df_filtered = filter_chicago_data_polars(df_processed)
        monthly_avg = compute_monthly_average_price_polars(df_filtered)
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        # Test model evaluation
        df_processed = preprocess_data_polars(self.df.clone())  # Ensure preprocessing
        df_filtered = filter_chicago_data_polars(df_processed)
        monthly_avg = compute_monthly_average_price_polars(df_filtered)
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        mae, r2 = evaluate_model(model, X, y)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(r2, float)


if __name__ == "__main__":
    unittest.main()
