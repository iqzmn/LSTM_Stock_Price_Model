# The LSTM RNN stock prices model

[**Recurrent neural networks (RNNs)**](https://en.wikipedia.org/wiki/Recurrent_neural_network) are a class of artificial neural networks for sequential data processing. One of them is LSTM (Long Short-Term Memory) network.

[**The Moscow Exchange (MOEX)**](https://www.moex.com/en) is the largest exchange in Russia, operating trading markets in equities, bonds, derivatives, the foreign exchange market, money markets, and precious metals. In the Main Market sector more than 1,400 securities from roughly 700 Russian issuers are available for trading every day.

The list of securities can be found here: https://www.moex.com/en/moexboard/instruments-list.aspx

## Model further improvements:
1. Finding model parameters using: Grid Search, Random Search, Bayesian optimization, and cross-validation to prevent overfitting
2. Using several securities as historical data for training
3. Using not one attribute (price), but several (volume, high, low prices, etc.) + corr matrix
4. To add weekly, monthly data for training and prediction
5. To add sentiment analysis