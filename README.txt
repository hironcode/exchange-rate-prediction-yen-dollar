These files were created for my personal research use.
The Prophet's prediction consists of three parts: US dollar interest rates, Japanese yen interest rates, and yen/dollar exchange rates.
The purpose is to predict the future exchange rates based on the two interest rates.
The reason for choosing the interest rates as regressors is to see how the correlation between interest rates and exchange rates
clarified in the interest parity holds in the prediction.

Any functions or equivalent sets of codes that I cited from external sources are not my work.
Please see REFERENCES.txt to find resources I have referred to or cited.

Data:
I used the following data for the analysis
(1) dollar interest rate CSV file from FRED Economic Data (1970-01 through 2023-11)
    https://fred.stlouisfed.org/series/FEDFUNDS
(2) yen interest rate CSV file from Bank of Japan (1993/10 through 2023/9)
    https://www.stat-search.boj.or.jp/index_en.html#
(3) yen/dollar exchange rate CSV file (1973/01 through 2023/10)
    https://www.stat-search.boj.or.jp/index_en.html#

Please make a "data" folder in the same directory as main.ipynb and place your downloaded data inside.
