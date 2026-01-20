# Are Stocks Normal?

An interactive Streamlit app that investigates whether stock market returns follow a normal distribution (bell curve). Many financial models assume normality. This tool helps you see when that assumption breaks down.

## Why This Matters

Financial models like Black-Scholes assume stock returns are normally distributed. In reality, markets exhibit:

- **Negative skew.** Panic selling creates longer left tails.
- **Fat tails (excess kurtosis).** Extreme moves happen more often than a bell curve predicts.

Understanding these deviations is critical for risk management. On Black Monday (October 19, 1987), the S&P 500 dropped 23% in a single day. Under normal distribution assumptions, this event would have near-zero probability.

## Features

- **Multiple assets.** S&P 500, QQQ, major tech stocks, financials, consumer staples, and crypto (BTC, ETH).
- **Flexible intervals.** Daily, weekly, monthly, or quarterly returns.
- **Custom date ranges.** Analyze any period from 1980 to present.
- **Visual analysis.** Histograms comparing observed data vs. a perfect bell curve.
- **Q-Q plots.** Quickly spot deviations from normality.
- **Statistical tests.** Kolmogorov-Smirnov and Shapiro-Wilk tests with interpretation.
- **Educational context.** Explanations of skewness, kurtosis, and real-world implications.

## Tech Stack

- **Data**: yfinance
- **Analysis**: pandas, numpy, scipy, statsmodels
- **Visualization**: matplotlib, Streamlit
- **Language**: Python

## Jupyter Notebook

A notebook version (`areStocksNormal.ipynb`) is also available for interactive exploration.
