import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from datetime import date
from scipy import stats
from scipy.special import erfinv

@st.cache_data
def fetch_stock_data(ticker, interval, start, end):
    raw = yf.download(ticker, interval=interval, start=start, end=end, auto_adjust=True)
    if len(raw) == 0:
        return pd.DataFrame()
    data = raw[['Close']].copy()
    data.columns = data.columns.droplevel(1)
    data['return'] = np.log(data['Close'] / data['Close'].shift(1))
    return data.iloc[1:].copy()

# TITLE
st.title('Do Stock Returns Follow a Normal Distribution (AKA Bell Curve🔔)?')
st.markdown('#')

# USER INPUT
col1, col2 = st.columns(2)
ui_ticker = col1.selectbox('Pick a stock ticker', 
                           ['^GSPC', 'QQQ', 'EFA', 
                            'MSFT', 'META', 'AAPL', 'NFLX', 'GOOG', 'ORCL', 'TSLA',
                            'JPM', 'BRK-A', 'KKR', 'V',
                            'COST', 'PEP', 'WMT',
                            'BTC-USD', 'ETH-USD'])

intervals = ['1d', '1wk', '1mo', '3mo']
ui_interval = col2.selectbox('Pick a time interval (nonoverlapping jumps)', intervals, index=1)

col1, col2, col3, col4 = st.columns(4)
curr_year = date.today().year
curr_month = date.today().month
months = list(range(1, 13))
years = list(range(1980, curr_year+1))
ui_start_month = col1.selectbox('Start Month of study', months, index=0)
ui_start_year = col2.selectbox('Start Year of study', years, index=0)
ui_end_month = col3.selectbox('End Month of study', months, index=curr_month-1)
ui_end_year = col4.selectbox('End Year of study', years, index=len(years)-1)
ui_start = date(ui_start_year, ui_start_month, 1)
ui_end = date(ui_end_year, ui_end_month, 28)

# PULL DATA
data = fetch_stock_data(ui_ticker, ui_interval, ui_start, ui_end)

if len(data) > 30:
    valid = True
else:
    valid = False
    st.write("There are less than 30 data points. This is insufficient to complete the analysis.")
    st.write("Please adjust the start date, end date and/or interval and try again.")

if valid:
    # PERFORM ANALYSIS
    mu = data['return'].mean()
    biggest_gain = data['return'].max()
    biggest_loss = data['return'].min()
    biggest_gain_date = data['return'].idxmax()
    biggest_loss_date = data['return'].idxmin()

    n = len(data)
    var = np.power(data['return'] - mu, 2).sum()/(n-1)
    std = np.sqrt(var)

    skew = np.power((data['return'] - mu)/std, 3).sum()*n/((n-1)*(n-2))
    if skew < -0.5:
        skew_msg = ('the distribution has a long left tail because people were panic selling!📉🫨 '
        'This indicates the returns do not follow a normal distribution')
    elif skew < 0.5:
        skew_msg = ('the distribution is relatively symmetric '
        'This indicates the returns follow a normal distribution')
    else:
        skew_msg = ('the distribution has a long right tail because people are euphoric buying!🚀 '
        'This indicates the returns do not follow a normal distribution')

    ex_kurt = np.power((data['return'] - mu)/std, 4).sum()*n*(n+1)/((n-1)*(n-2)*(n-3)) - 3*(n-1)**2 / ((n-2)*(n-3))
    if ex_kurt < -0.5:
        kurt_msg = ('the distribution has light 🐥 tails because this stock is boring and doesnt move much. '
        'This indicates the returns do not follow a normal distribution')
    elif ex_kurt < 0.5:
        kurt_msg = ('the distribution has "normal" weight in the tails '
        'This indicates the returns follow a normal distribution')
    else:
        kurt_msg = ('the distribution has heavy 🦍 tails because the stock has extreme price moves. '
        'This is likely due to earnings reports and investors reacting to breaking news! '
        'This indicates the returns do not follow a normal distribution')

    # DISPLAY RESULTS

    # Data Summary
    st.markdown('#')
    st.write('Quick Look at Data Pulled for Study')
    data_summary = pd.DataFrame({
        'description': ['Sample Size', 
                        'Interval of Returns',
                        'First Data Point', 
                        'Last Data Point', 
                        f'Bigest {ui_interval} Gain Start Date', 
                        f'Bigest {ui_interval} Gain', 
                        f'Biggest {ui_interval} Loss Start Date',
                        f'Biggest {ui_interval} Loss',
                        'Average Return',
                        'Standard Deviation of Return'
                        ],
        'value': [str(n), 
                  ui_interval,
                  str(data.index[0].date()),
                  str(data.index[-1].date()),
                  str(biggest_gain_date.date()),
                  f'{biggest_gain:.2%}',
                  str(biggest_loss_date.date()),
                  f'{biggest_loss:.2%}',
                  f'{mu:.2%}',
                  f'{std:.2%}'
                  ]
        })
    data_summary.set_index('description', inplace=True)
    st.table(data_summary)


    #  Histogram Look
    st.markdown('#')
    st.subheader('Do the returns look like a bell curve?')
    st.text('Note that a bell curve should be symmetric.\n'
            'We can measure symmetry with skewness which is 0 for a symmetric distribution.\n\n'
            f'The skew for the observed data is {skew:.2f} which means {skew_msg}')
    st.text('')
    st.text('Note that a bell curve should have a medium amount of weight in its tails.\n'
            'This means that stocks have extreme moves occasionally but not all the time\n'
            'We can measure tail weight with excess kurtosis which is 0 for a normal distribution.\n\n'
            f'The excess kurtosis for the observed data is {ex_kurt:.2f} which means {kurt_msg}')

    perfectly_normal_data = mu + std * np.sqrt(2) * erfinv(2 * (np.arange(1, n + 1) - 0.5) / n - 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data['return'])
    axes[0].set_title("Observed Data")

    axes[1].hist(perfectly_normal_data)
    axes[1].set_title("Perfectly Normal Bell Curve")

    for ax in axes:
        ax.set_xlabel(ui_interval + " stock returns")
        ax.set_ylabel("number of returns in bucket")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    st.pyplot(fig)


    # Q Q Plot
    st.markdown('#')
    st.subheader('Now let\'s look at a Q-Q plot')
    data['s_return'] = (data['return'] - mu) / std
    fig = plt.figure()
    sm.qqplot(data['s_return'], line='45', ax=fig.add_subplot(111))
    st.pyplot(fig) 
    
    st.text(
            'What is a Q-Q plot though?!?\n\n'
            'In Short:\n'
            'If the data follow a 45° line, that\'s evidence that'
            ' the data follow a normal distribution. '
            'Otherwise we have evidence that the distribution does not follow a normal distribution.\n\n'
            'In detail:\n'
            'A Q-Q plot is a plot of theoretical quantiles pulled from a perfectly normal distribution'
            ' plotted against quantiles from the observed distribution after standardization. '
            'Standardization subtracts the mean and divides by the standard deviation. '
            'This makes it so each return is expressed in how many standard deviations it is away from the mean. '
            'Standardization makes it so when we plot these against each other '
            'they should follow a 45° line (y=x). '
            'Note that for a normal distribution 99.7% of the data should '
            'fall within 3 standard deviations of the mean '
            'so data outside of that range are considered outliers. '
            'Seeing many outliers will be reflected by high excess kurtosis '
            'and indicates that the distribution is not normal.'
            )

    # Formal Normality Tests
    st.markdown('#')
    st.subheader('Can we do a hypothesis test?')
    st.text(
            'In Short:\n'
            'Yes, there are several hypothesis tests that exist to test for normality. '
            'However, they are highly sensitive to large sample sizes. '
            'If sample size is large there is a tendency to always reject the null hypothesis that '
            'the data comes from a normal distribution.\n'
            'Thus, they should be thought of as another tool in the toolbox rather than a source of truth.\n\n'
            
            'In Detail:\n'
            'Typically, when we setup a hypothesis test, we want to reject the null hypothesis. '
            'This is because typically the null hypothesis is that there is no difference in things we are studying, '
            'and we want to show that there IS a difference in the things we are studying.\n'
            'As we add more data to the study, '
            'we have more evidence to make estimates, so they stabilize as variance of the estimate decreases. '
            'As a result, adding more data helps us achieve our goal of trying to '
            'show a difference in the things we are studying.\n\n'

            'On the contrary, the null hypothesis to test for normality '
            'is that the observed data does follow a normal a distribution. ' 
            'So here we are hoping to fail to reject the null hypothesis. '
            'As our sample size increases, we become more likely to reject the null hypothesis because '
            'variance decreases and even small deviances from normality will cause us to reject the null hypothesis.'
            )

    def get_p_msg(p_value):
        if p_value < 0.01:
            return ('We have strong statistical evidence to reject the idea that '
                    'the stock returns come from a normal distribution.')
        elif p_value < 0.05:
            return ('We have some statistical evidence to reject the idea that '
                    'data come from a normal distribution.')
        else:
            return ('We have statistical evidence to fail to reject the idea that '
                    'the data come from a normal distribution')


    ks_stat, ks_p = stats.kstest(data['s_return'], 'norm')
    sh_stat, sh_p = stats.shapiro(data['s_return'])

    st.subheader('Okay, Let\'s see the results')
    st.text(
            f'Sample Size for Reference: {n}\n\n'
            'Kolmogorov Smirnov Test for Normality:\n'
            'Compares the observed cumulative distribution of the data with '
            'the cumulative distribution for a normal distribution. '
            'Large deviations indicate non-normality.\n\n'
            'The null hypothesis is that the observed data follows a normal distribution.\n'
            f'p-value: {ks_p:.10f}\n'
            f'{get_p_msg(ks_p)}\n\n'
            'Shapiro Wilk Test for Normality:\n'
            'Compares order statistics from observed data with order statistics from a normal distribution\n\n'
            'The null hypothesis is that the observed data follows a normal distribution.\n'
            f'p-value: {sh_p:.10f}\n'
            f'{get_p_msg(sh_p)}'
            )

    # Importance
    st.markdown('#')
    st.subheader('Why is this important?')
    st.markdown('''
                In the financial industry it often useful to assume stock returns follow a normal distribution. 
                So having an in depth understanding of this assumption is critical. 
                Assuming normality often makes modeling stock returns feasible, 
                but it should be understood that in reality 
                stock returns tend to have **negative skew** from panic sellers 🫨📉 
                and **fat tails** from breaking news, earnings reports, panic selling and FOMO buying.\n

                Here are a few specific examples of real-world implications:

                ### Black-Sholes Option Pricing and Greeks Calculation ###
                The Black-Scholes model and the differential equations used to calculate option Greeks 
                assume that stock returns follow a normal distribution. 
                Violations of this assumption can lead to mispricing, 
                especially for options far from the money (because the tails become more important!).

                ### TVaR and Risk Management ###
                **Tail Value at Risk (TVaR)** measures expected losses in extreme events 
                (i.e., given that losses exceed a certain threshold).
                If we assume normality when in reality our tails of our distribution are different, 
                our TVaR will be wildly inaccurate and leaves us out in the cold when unfortunate events occur.
                To overcome this we may assume normality when we care about the middle of the distribution 
                but use empircal driven assumptions and professional judgement to handle risk relating to tails 
                such as TVaR.

                ### Black Monday October 19th 1987 ###
                The 1987 stock market crash, known as **Black Monday**, 
                saw the S&P500 drop about 23% in a single day! (You can see this as the biggest drop in data above).
                A contributing factor was the widespread use and 'abuse' of the Black-Scholes option pricing model.

                Traders relied on Black-Scholes to price options and hedge positions dynamically. 
                Many sold out-of-the-money options, 
                thinking the probability of a large loss was negligible.
                In the words of Nassim Taleb, it was like “picking up pennies in front of a steamroller”.

                When markets fell sharply, dynamic hedging required selling to remain delta-neutral, 
                which amplified the crash. 
                On the other hand, 
                firms that recognized the model’s limitations profited by purchasing OTM puts as insurance.

                This event illustrates how assuming normality and ignoring tail risks can lead to
                serious losses of real money.
                ''')

    

