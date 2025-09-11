[![Avocado Price Analysis](https://github.com/ammylin/avocado-price-analysis/actions/workflows/main.yml/badge.svg)](https://github.com/ammylin/avocado-price-analysis/actions/workflows/main.yml)

# IDS 706: Avocado Price Analysis (Week 2)

## Project Description
The provided code is a comprehensive analysis of avocado prices in Chicago, utilizing both Pandas and Polars libraries for data manipulation and analysis. The goal is to determine how average avocado prices vary by month in the year 2015. The analysis employs linear regression to model the relationship between the month and the average price of avocados.

### Setup Instructions 
To set up this repository, I followed the steps outlined in Week 1's Python template here: `https://github.com/ammylin/IDS_706_python_temp`. Additionally, I loaded the CSV file from Kaggle in order to perform analysis on it. I also installed the required libraries, which I outline below: 

#### Download the Dataset
1. Go to the Kaggle dataset page: [Avocado Prices Dataset](https://www.kaggle.com/datasets/neuromusic/avocado-prices/data).
2. Click on the "Download" button to download the dataset.
3. Move the downloaded CSV file (`avocado.csv`) to the project directory where the code is located.

#### Required Libraries
- Pandas (used for data analysis)
- Polars (used for data analysis)
- Matplotlib (used to make visualizations)
- scikit-learn (used for the machine learning algorithm, in which I used LinearRegression)

#### Installation
To install the required libraries, you can use pip. Run the following command in your terminal:

```
pip install pandas polars matplotlib scikit-learn
```

## Key Components of the Code
### Data Loading and Exploration
The code includes functions to load the avocado dataset using both Pandas and Polars. I inspected the data during my exploratory data analysis (EDA) using functions like `.head()`, `.info`, etc., which I did not include in my file. 

### Data Preprocessing
Data preprocessing is crucial for preparing the dataset for analysis. The code includes separate functions for both libraries to:
- Remove duplicates.
- Convert the "Date" column to a datetime format.
- Extract the month from the date.

### Data Filtering
The analysis focuses specifically on avocados sold in Chicago. Functions like `.groupby`/`.group_by` and `.filter` are utilized to filter the dataset accordingly.

### Monthly Average Price Calculation
The average price of avocados is computed by month using both libraries, allowing for a direct comparison of performance.

### Model Training and Evaluation
A linear regression model is trained using the monthly average prices, and its performance is evaluated using metrics such as Mean Absolute Error (MAE) and R-squared (R²).

### Visualization
The results are visualized using Matplotlib, showing the actual vs. predicted prices, which helps in understanding the model's performance.

### Test File 
I created a test file, `test_avocado_analysis.py`, that test whether `avocado_analysis.py` properly loads, preprocesses, filters, and runs the ML algorithm and analysis correctly. 
To ensure the code functions correctly, run the test file `test_avocado_analysis.py`. This file tests the following:
- Data loading and preprocessing.
- Filtering for Chicago data.
- Monthly average price calculations.
- Model training and evaluation.

### Outcomes 
#### Pandas Analysis
![Avocado Prices in Chicago Based on Month (Pandas)](avg_prices_pandas.png)

Using Pandas, I got the following metrics:
- Mean Absolute Error (MAE): `0.12` (This indicates the average difference between predicted and actual prices.)
- R-squared: `0.15` (This means that 15% of the variability in avocado prices is explained by the model.)

#### Polars Analysis
![Avocado Prices in Chicago Based on Month (Polars)](avg_prices_polars.png)

Using Polars, I got the following metrics:
- Mean Absolute Error (MAE): `0.02`
- R-squared: `0.97`

#### Summary
All in all, it appears that Polars provides a better regression model for the relationship between the month and the average price of avocados. We can see this because the Mean Absolute Error (MAE) using Polars is `0.02`, which means that the predicted average price of avocados is off by $0.02 from the actual prices, while it is `0.12` for the Pandas analysis. Likewise, the R-squared value for the Polars analysis is `0.97`, which means that 97% of the variability in avocado prices can be explained by the model. This indicates a strong fit, suggesting that the month is a significant predictor of avocado prices in Chicago.

In contrast, the Pandas analysis, with an R-squared value of `0.15`, shows that only 15% of the variability in avocado prices is accounted for by the model, indicating a much weaker relationship. This stark difference in performance highlights the effectiveness of Polars in handling this regression task, likely due to its optimized performance and efficient data handling capabilities.

Overall, the results suggest that for this type of analysis, using Polars may yield more accurate and reliable predictions compared to Pandas. Likewise, the models would be more accurate if trained on more data. 

### Data Source & Acknowledgments 
This dataset was retrieved from a public dataset on Kaggle linked here: `https://www.kaggle.com/datasets/neuromusic/avocado-prices/data`. These data were originally downloaded from the Hass Avocado Board website (`http://www.hassavocadoboard.com/retail/volume-and-price-data`) in May of 2018 and compiled into a single CSV. 

I used the Python template from Week 1 of this class, which referred to Professor Yu's template and instructions. Additionally, some of the code for this project was generated using the assistance of generative AI (specifically, OpenAI's GPT-4o and GitHub Copilot). 