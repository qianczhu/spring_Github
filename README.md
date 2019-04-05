# Data Science Projects
## Data Wrangling, inferential statistics and machine learning & deep learning projects
Python codes for the mini projects can be found in the folder. 
## Capstone Project 1: Earnings Surprise Predictions using Machine Learning
### Summary
#### Problem statement
To analyse a combination of financial data to predict stock movements after the release of earnings reports. 

_Intended use_:
* incorporate predictions from this project into fund management strategies 
* to protect against wild stock fluctuations during reporting season
* to plan for investment strategies accordingly

#### The path leading to the solution
* Gather data
* Data wrangling
* Exploratory data analysis
* Machine learning

#### Feature engineering
Example: Raw data contains trader mood message volume and sentiment data. The transformed data contains sum of total message volumes for 5 days prior, as well as weighted sum of sentiment at date-0, date-1, date-2, date-3, date-4 (prior), with increasing weight approaching event date. 

#### Statistical analysis
Correlation matrix for selective features variables:
![Correlation matrix](/project1_corr_matrix.png)

#### Machine learning
##### Cross Validation (CV) for time series data
Time series data requires a different CV to traditional techniques such as K-fold validation. Traditional techniques randomly splitting the data into training and test/validation sets, which causes data leakage. 

This project: split point is made at the 80% mark of the ordered list of observations.

##### Evaluation matrics
Traditional metrics such as _Accuracy_ would not work well here. In this project, _scoring = 'neg_log_loss'_ was chosen.

_Reason_:
* to penalise predictions which are highly confident (high probability) but wrong.
* prefer to be correct and less confident on predicting a stock than vice versa. 

##### Models implemented
* Logistic regression
* Random forest
* AdaBoost
* Support Vector Machine (SVM)
* XGBoost

For all models, hyperparameter tunings were performed by using RandomizedSearchCV, to search for the optimised parameters for that particular model. 

##### A trading strategy
Top stocks with predicted label “+1”, ranked by probability, for 80-day drift:
![results table](/results_table.png)

Buy the top predicted stocks and hold for 80 days on the second day after earnings report release.

## Capstone Project 2: Prediction of cryptocurrency movements using deep learning
### Summary
#### Problem statement
Cryptocurrencies are a relatively new type of financial instrument, which has sparked new interest in both the public and financial domains recently. Price movements of cryptocurrencies can sometimes be volatile, which poses a challenge for investors. This project aims to use deep learning to make predictions for the top cryptocurrencies, based on a combination of price data and fundamentals data. The prediction will be in the form of direction of price movement, i.e., up or down for a number of time intervals.

#### Data acquisition
This project leverages data from a number of sources to predict the movements of major cryptocurrencies such as Bitcoin, Ethereum, Ripple, etc. The data sources consist of the following:
* Cryptocurrency prices from: 
    * Quandl API
    * CryptoCompare API (prices and volumes)
* Developers and social data:
    * More Blockchain data on Quandl, including Bitcoin My Wallet Users, Bitcoin difficulty, Bitcoin Miners Revenue, etc. (Quandl API)
    * CoinGecko

Both daily data and hourly data were acquired. The hourly data was later aggregated to form 4-hourly, 8-hourly, 12-hourly data for experimentation.

#### Data cleaning and wrangling
Data from different sources were merged and aggregated. Missing values were dealt with. To filter out the high frequency noise, Wavelet Transform was used (Python _pywt_ package): 

_Wavelet figure

The data shown above was for the number of transactions made by My Wallet Users per day. The original data contains arbitrary high frequency noise---after being filtered out, we are left with a smoother curve that still captures major disruptions (which may correspond to events) in the data.  

#### Exploratory data analysis
##### Correlation analysis
The raw data from Quandl and GoinGecko have variables that are highly correlated with each other. This is illustrated in the correlation matrix plot below. Columns which have correlation of greater than 0.9 with BTCUSD and some other column, were dropped. 

Fig. correlation matrix plot for raw data from Quandl and CoinGecko. 

It is well known that the cryptocurrency markets generally move together. In order to examine how valid this statement is, the correlation matrix was calculated and plotted. It can be seen that most of the cryptocurrencies move in sync with each other except very few, e.g., USDT (which is tied to the US dollar), SHND, ZEC and CPC. 

Fig. Correlation matrix of various cryptocurrencies. 

##### ARIMA
ARIMA is a commonly used statistical forecasting method for time series data. In ARIMA, future values of a variable is assumed to be a linear function of several past observations and random errors. In this project, ARIMA can be incorporated into the methodology to address the issues of seasonality and non-stationary data. The figure below shows the rolling mean and standard deviation for the LTC closing prices, which clearly demonstrates that the data is non-stationary. 

Fig. Rolling mean and standard deviation for the LTC closing prices (raw data). 

Differencing was implemented to remove the trend and seasonality. Fig. 6 shows that the data is now stationary. Subsequently, Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) were plotted to determine the p and q parameters of the ARIMA(p,d,q) model. The LTC_7day_return and ARIMA predicted LTC price are plotted in Fig. 7. The ARIMA predicted price was hence incorporated in the feature variables for the LSTM model.

Fig. 6. Rolling mean and standard deviation for the LTC closing prices (after removing the trend and seasonality).

Fig. 7. LTC_7day_return and ARIMA predicted LTC price. 

##### Log returns

#### Machine learning techniques

#### Pairs trading

#### Deep learning results

#### Trading strategy

