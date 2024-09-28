# LSTM_Stock_Price

The LSTM Recurrent Neural Network to predict stock prices on MOEX.

**The Moscow Exchange (MOEX)** is the largest exchange in Russia, operating trading markets in equities, bonds, derivatives, the foreign exchange market, money markets, and precious metals. In the Main Market sector more than 1,400 securities from roughly 700 Russian issuers are available for trading every day.

The list of securities can be found here: https://www.moex.com/en/moexboard/instruments-list.aspx

## Model further improvements:
1. Finding model parameters using: Grid Search, Random Search, Bayesian optimization, and cross-validation to prevent overfitting
2. Use not one, but several securities as historical data for training
3. Use not one attribute (price), but several (volume, high, low prices, etc.) + corr matrix
4. Add weekly, monthly data for training and prediction
5. Add sentiment analysis