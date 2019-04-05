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
### Report Excerpt
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
Raw time series price data generally changes very abruptly, as shown in Fig. 8. By converting the price into its log form, the changes are more gradual, which may help the analysis later on. Also, since the returns are not normally distributed (in fact, they are lognormally distributed), by converting the returns into Log, we will obtain normally distributed values (i.e., taking log of lognormal would give us normally distributed values). Fig. 9 shows the Log Returns of different time periods. As we can see, different periodicity is present for different time frames.  

Fig. 8. Raw BTC/USD prices and its log prices. 

Fig. 9. Log of (a) daily return, (b) weekly return, (c) 30-day return and (d) 60-day return, for BTC/USD. 

#### Machine learning techniques
##### Feature engineering
Since we will be using previous days data features/variables to make predictions for future returns, the relevant features were shifted by N-days. The problem is chosen as a classification one, i.e., to predict the direction of the price movement after N-days: 1 for up, and 0 for down. Labels for binary classification were created for this purpose. The statistical procedure __Principal Component Analysis (PCA)__ was used to convert these possibly correlated variables into a set of linearly uncorrelated variables. After fitting the PCA model, 26 principal components were retained after the original 60 columns, while still maintaining 95% of the original characteristics of the data. 

Standardisation was applied to the numeric data. Specifically, the MinMaxScaler was chosen, which shrinks the range such that the range is between 0 and 1. This scaler works better for cases where the StandardScaler might not work so well. If the distribution is not Gaussian or the standard deviation is very small, the MinMaxScaler works better.

#### Pairs trading
Pairs trading strategy, which makes use of two cointegrated stocks (or ETFs, currencies or commodities, etc.), has a long history in the investment realm. In this project, the pairs trading strategy was combined with Machine Learning to extract useful information for the cryptocurrencies. This has not been reported before in the literature. 

#### Deep learning results
The Deep Learning part of this project is implemented on Google Colab (which runs on the Cloud and allows access to Google’s powerful computing resources such as GPU). Python’s Deep Learning library Keras with TensorFlow backend was used. 

Different cryptocurrencies and different time frames were investigated. An example is predicting the Litecoin weekly returns. Fig. 12 shows the comparison of counts between positive returns (class 1) and negative returns (class 0). The occurrence of the two classes is fairly similar; we don’t have an imbalanced class problem.  Fig. 13 shows the distribution of weekly returns for both the training data and test data. As expected, training data contains a wider range of returns (from -0.75 to 1.25, N.B. these are in Log returns) than the test data. This is advantageous for the Deep Learning model, as the model has more extreme cases to learn from. 

Fig. 12. Counts of positive weekly returns (class 1) and negative weekly returns (class 0), for Litecoin. 

Fig. 13. Distribution of weekly returns for Litecoin: (a) training data; (b) test data.

In contrast to traditional Machine Learning algorithms, Long Short Term Memory (LSTM, one of the Recurrent Neural Networks) takes into account the sequential nature of the time series data. In this project, LSTM was used exclusively to analyse the processed time series data mentioned above. 

Fig. 14 shows the weighted_avg scores for the test data, over 8 different training sessions. Overall, the training accuracy various slightly between different sessions, with a mean f1-score, precision and recall all equal to 0.6. This indicates that the model is quite stable and well-balanced, i.e., the f1-score, precision and recall values are almost identical. 

Fig. 14. Validation scores for the test data for the daily dataset.

As for the 12-hourly dataset, we have twice as much data, which is beneficial for the model. Fig. 16 shows the individual scores obtained for the test data, over 8 training sessions. The mean score is slightly worse than the daily dataset. Fig. 17 presents confusion matrix and classification report for one of the training instances. Again, the results are quite balanced, with slightly better accuracy in predicting for class 0, which is the negative returns. Fig. 18 shows the overall picture of overlapping the correct predictions with the original raw test data. The model excels in making predictions when positive return > 0.1 and negative return > 0.2. This is similar with the daily dataset, which confirms that the model can make predictions for extreme cases---this is where potential large profits or losses occur. 

Fig. 16. Validation scores for the test data for the 12-hourly dataset.

Fig. 18. Correct predictions vs the raw labeled test data for the 12-hourly dataset.

#### Trading strategy
According to the timeframe available (e.g., daily or 12-hourly), enter: 
(a) a long position if model prediction = 1, or 
(b) a short position if model prediction = 0. 

While holding a position: 
* if the subsequent model prediction changes direction, exit the position;
* if the subsequent model prediction remains in the same direction, either hold the position or place an extra trade. 
