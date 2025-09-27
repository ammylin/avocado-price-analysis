import unittest
import pandas as pd
import polars as pl
from avocado_analysis import (
    load_data,
    preprocess_data,
    filter_chicago_data,
    compute_monthly_average_price,
    train_model,
    evaluate_model,
)


class TestAvocadoModelPandas(unittest.TestCase):
    """Test case for the avocado analysis using Pandas."""

    def setUp(self):
        data = {
            "Date": [
                "2021-01-01",
                "2021-01-08",
                "2021-01-15",
                "2021-01-22",
                "2021-01-29",
            ],
            "AveragePrice": [1.00, 1.10, 1.20, 1.30, 1.40],
            "region": ["Chicago"] * 5,
        }
        self.df = pd.DataFrame(data)

    def test_load_data(self):
        df = load_data("avocado.csv", library="pandas")
        self.assertIsNotNone(df)
        self.assertGreater(df.shape[0], 0)
        self.assertEqual(df.shape[1], 14)

    def test_preprocess_data(self):
        df_processed = preprocess_data(self.df.copy(), library="pandas")
        self.assertEqual(df_processed.shape[0], 5)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_processed["Date"]))
        self.assertIn("month", df_processed.columns)

    def test_filter_chicago_data(self):
        df_filtered = filter_chicago_data(self.df.copy(), library="pandas")
        self.assertEqual(df_filtered.shape[0], 5)
        self.assertTrue((df_filtered["region"] == "Chicago").all())

    def test_compute_monthly_average_price(self):
        df_processed = preprocess_data(self.df.copy(), library="pandas")
        df_filtered = filter_chicago_data(df_processed, library="pandas")
        monthly_avg = compute_monthly_average_price(df_filtered, library="pandas")
        self.assertEqual(monthly_avg.shape[0], 1)
        self.assertAlmostEqual(monthly_avg["AveragePrice"].iloc[0], 1.20)

    def test_train_and_evaluate_model(self):
        df_processed = preprocess_data(self.df.copy(), library="pandas")
        df_filtered = filter_chicago_data(df_processed, library="pandas")
        monthly_avg = compute_monthly_average_price(df_filtered, library="pandas")
        X = monthly_avg[["month"]]
        y = monthly_avg["AveragePrice"]
        model = train_model(X, y)
        self.assertIsNotNone(model)

        mae, r2 = evaluate_model(model, X, y)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(r2, float)


class TestAvocadoModelPolars(unittest.TestCase):
    """Test case for the avocado analysis using Polars."""

    def setUp(self):
        data = {
            "Date": [
                "2021-01-01",
                "2021-01-08",
                "2021-01-15",
                "2021-01-22",
                "2021-01-29",
            ],
            "AveragePrice": [1.00, 1.10, 1.20, 1.30, 1.40],
            "region": ["Chicago"] * 5,
        }

        self.df = pl.DataFrame(
            {
                "Date": pl.Series("Date", data["Date"], dtype=pl.Utf8),
                "AveragePrice": pl.Series(
                    "AveragePrice", data["AveragePrice"], dtype=pl.Float64
                ),
                "region": pl.Series("region", data["region"], dtype=pl.Utf8),
            }
        )

    def test_load_data(self):
        df = load_data("avocado.csv", library="polars")
        self.assertIsNotNone(df)
        self.assertGreater(df.height, 0)
        self.assertEqual(df.width, 14)

    def test_preprocess_data(self):
        df_processed = preprocess_data(self.df.clone(), library="polars")
        self.assertEqual(df_processed.height, 5)
        self.assertEqual(df_processed["Date"].dtype, pl.Date)
        self.assertIn("month", df_processed.columns)

    def test_filter_chicago_data(self):
        df_filtered = filter_chicago_data(self.df.clone(), library="polars")
        self.assertEqual(df_filtered.height, 5)
        self.assertTrue((df_filtered["region"] == "Chicago").all())

    def test_compute_monthly_average_price(self):
        df_processed = preprocess_data(self.df.clone(), library="polars")
        df_filtered = filter_chicago_data(df_processed, library="polars")
        monthly_avg = compute_monthly_average_price(df_filtered, library="polars")
        self.assertEqual(monthly_avg.height, 1)
        self.assertAlmostEqual(monthly_avg["AveragePrice"][0], 1.20)

    def test_train_and_evaluate_model(self):
        df_processed = preprocess_data(self.df.clone(), library="polars")
        df_filtered = filter_chicago_data(df_processed, library="polars")
        monthly_avg = compute_monthly_average_price(df_filtered, library="polars")
        X = monthly_avg.select("month").to_numpy().reshape(-1, 1)
        y = monthly_avg.select("AveragePrice").to_numpy().ravel()
        model = train_model(X, y)
        self.assertIsNotNone(model)

        mae, r2 = evaluate_model(model, X, y)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(r2, float)


if __name__ == "__main__":
    unittest.main()
