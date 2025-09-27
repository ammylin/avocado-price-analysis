[![Avocado Price Analysis](https://github.com/ammylin/avocado-price-analysis/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/ammylin/avocado-price-analysis/actions/workflows/main.yml)

# IDS 706: Avocado Price Analysis (Weeks 2 & 3)

## Project Description
The provided code is a comprehensive analysis of avocado prices in Chicago, utilizing both Pandas and Polars libraries for data manipulation and analysis. The goal is to determine how average avocado prices vary by month in the year 2015. The analysis employs linear regression to model the relationship between the month and the average price of avocados. 

### Setup Instructions 
To set up this repository, I followed the steps outlined in Week 1's Python template here: `https://github.com/ammylin/IDS_706_python_temp`. Additionally, I loaded the CSV file from Kaggle in order to perform analysis on it. I also installed the required libraries, which I outline below, as well as my instructions for setting up Dev containers and Docker: 

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

#### Creating a Dev Container and Configuring Docker
After downloading the Dev Containers extension from VSCode and installing the Docker desktop application, I created my Dev container in VSCode using the following steps: 
1. Press `Cmd + Shift + P`, then click on "Dev Containers: Add Dev Container Configuration Files". 
2. Select the configuration of choice (for this, I selected the Python 3 configuration). 
3. In the next menu, add more features as needed. 
4. Open the Dev container by clicking on the blue button on the bottom left corner of VSCode. 

At this point, we can open up the Docker desktop app and see, under "Containers", the details of our configuration. Now, to build an image from the Dockerfile and run the Docker container, I created a Dockerfile and related setups by: 
<<<<<<< HEAD
1. Press `Cmd + Shift + P`, then click on "Docker: Add Docker Files to Workspace." 
=======
1. Press `Cmd + Shift + P`, then click on "Containers: Add Docker Files to Workspace." 
>>>>>>> c42bf71 (refactoring)
2. In the terminal, I ran: `docker build -t container-name .` (For me, "container-name" was "naughty agnesi"). This builds an image from the Dockerfile. 
3. Next, I ran `docker run -d -p 8088:3000 --name my-avocado-container welcome-to-docker`. This runs the container in the background (-d), maps port 8088 on my host to port 3000 inside the container, and names the container my-avocado-container.
4. On Docker desktop app, under "Ports," I clicked `8088:3000`, which opened the following on my browser: 
![Congratuations! You ran your first container!](ran_docker.png)
This is what my Docker desktop app displayed under the "Containers" tab: 
![Docker Containers](docker_desktop.png)

Setting up Dev containers with Docker helps us address the "it works on my machine" problem; that is, by containerizing our development environment, we ensure consistent setups across different machines and team members. This makes onboarding smoother, eliminates environment mismatches, and streamlines deployment, since the container encapsulates all dependencies and configurations in a portable way.

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