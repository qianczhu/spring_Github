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


