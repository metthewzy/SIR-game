import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as datetime
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

oxford_file_path1 = 'COVIDandSTRINGENCY_data/NorwaySwedenStringency.csv'
ConfirmFile = 'COVIDandSTRINGENCY_data/NS_cases_confirmed.csv'
DeathFile = 'COVIDandSTRINGENCY_data/NS_deaths_confirmed.csv'


def read_NS_cases(start_date, end_date):
    dft = pd.read_csv(ConfirmFile)
    start_index = dft.columns.get_loc(start_date) - 1
    end_index = dft.columns.get_loc(end_date) + 1
    df = dft.iloc[:, start_index:end_index]
    days = list(dft.columns)
    days = days[days.index(start_date):days.index(end_date) + 1]
    # confirmed = df.sum()
    # df = dft[:,loc[start_date:end_date]]
    # df2 = pd.read_csv(DeathFile)
    # lenS = len(days)
    # confirmed = [0] * lenS
    # death = [0] * lenS
    # for s in statesI :
    # 	confirmedState = df[df.iloc[:, 0] == s]
    # 	confirmedState = confirmedState.iloc[0]
    # 	confirmed = [confirmed[i] + confirmedState[i] for i in range(lenS) ]
    # 	# death = death +  df2[df2.iloc[:, 0] == s]
    # death = death.iloc[0].loc[start_date: end_date]
    return df, days


def create_Rec_display():
    ret_series = []

    index = pd.date_range('2021-01-01', '2021-06-09')
    seriesR = pd.Series(0, index)
    ret_series.append(seriesR)

    index = pd.date_range('2021-06-10', '2021-07-26')
    seriesR = pd.Series(300000, index)
    ret_series.append(seriesR)

    index = pd.date_range('2021-07-27', '2021-08-19')
    seriesR = pd.Series(160000, index)
    ret_series.append(seriesR)

    index = pd.date_range('2021-08-20', '2021-09-12')
    seriesR = pd.Series(223000, index)
    ret_series.append(seriesR)

    index = pd.date_range('2021-09-13', '2021-10-12')
    seriesR = pd.Series(117000, index)
    ret_series.append(seriesR)

    index = pd.date_range('2021-10-13', '2021-11-16')
    seriesR = pd.Series(131000, index)
    ret_series.append(seriesR)

    index = pd.date_range('2021-11-17', '2021-12-10')
    seriesR = pd.Series(150000, index)
    ret_series.append(seriesR)

    index = pd.date_range('2021-12-11', '2022-01-08')
    seriesR = pd.Series(1300000, index)
    ret_series.append(seriesR)

    index = pd.date_range('2022-01-09', '2022-04-01')
    seriesR = pd.Series(890000, index)
    ret_series.append(seriesR)

    return ret_series


def read_oxford_data(confirmed, days):
    df1 = pd.read_csv(oxford_file_path1)
    start_date = "1-Feb-20"
    end_date = "1-Aug-20"
    start_index = df1.columns.get_loc(start_date)
    end_index = df1.columns.get_loc(end_date)+1
    df = df1.iloc[:, start_index:end_index]
    #	mask = (df['Date'].strptime("%d-%B-%y")).strftime("%y%m%d") > 20220401)
    # Norway_str =
    confirmed_Norway = confirmed.iloc[0]
    confirmed_Sweden = confirmed.iloc[1]
    stringency_Norway = df.iloc[0]
    stringency_Sweden = df.iloc[1]


    fig, ax = plt.subplots()
    colorS1 = 'lightgreen'
    colorS2 = 'lightblue'
    confirmed7_Norway = confirmed_Norway.rolling(7, min_periods=1).mean()
    confirmed7_Sweden = confirmed_Sweden.rolling(7, min_periods=1).mean()
    # dconfirmed_N = pd.Series(np.diff(confirmed))
    dconfirmedN7 = pd.Series(np.diff(confirmed7_Norway))/5370
    dconfirmedS7 = pd.Series(np.diff(confirmed7_Sweden))/10370
    first_legend = True


    # X = dfstate['Date']
    # X = [str(d) for d in X]
    # X = [d[0:4] + '-' + d[4:6] + '-' + d[6:8] for d in X]
    # Xdates = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in X]

    # Plot the two columns

    # ax2= ax.twinx()
    # ax2.plot(Xdates, dconfirmed, color='red')
    # Set the title and labels of the axes
    # ax.set_title('Data Frame Plot')

    # Show the plot

    # print(s)
    # dconfirmed =
    # ax.plot(Xdates, dconfirmed, color='red')

    ax.plot(days, stringency_Norway, color=colorS1, label='NorwayStringency')
    ax.plot(days, stringency_Sweden, color=colorS2, label='SwedenStringency')
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()

    # first_legend = True
    # for i in range(1, len(dconfirmed7A) - 1):
    #     if dconfirmed7A[i - 1] < 50000 <= dconfirmed7A[i] <= dconfirmed7A[i + 1]:
    #         if first_legend:
    #             first_legend = False
    #             ax2.axvline(Xdates[i], color='red', alpha=1, linestyle=':', label='Rising 50K Cases(7-day AVG)')
    #         else:
    #             ax2.axvline(Xdates[i], color='red', alpha=1, linestyle=':')
    #         print(Xdates[i].strftime('%Y-%m-%d'), 'stringency:', Y.iloc[i])

    # ax2.plot(days, dconfirmedN7, color='red', label='Norway Daily Cases(7-day AVG) per population')
    # ax2.plot(days, dconfirmedS7, color='blue', label='Sweden Daily Cases(7-day AVG) per population')

    ax2.plot(days,(confirmed7_Norway[1:])/5.37 , color='green', label='Norway Cumulative Cases(7-day AVG)')
    ax2.plot(days,(confirmed7_Sweden[1:])/10.37 , color='blue', label='Sweden Cumulative Cases(7-day AVG)')

    # plt.ticklabel_format(style='plain', axis='y')
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2.set_ylim(0, ax2.get_ylim()[1])
    ax.set_xlim(days[0],days[-1])
    ax.legend(loc='lower left', bbox_to_anchor=(-0.12, 1.02))
    ax2.legend(loc='lower right', bbox_to_anchor=(1.2, 1.02))
    fig.autofmt_xdate()
    ax.set_ylabel('Policy Stringency Index')
    ax2.set_ylabel('Daily Cases')

    # plt.show()
    fig.savefig('COVIDandSTRINGENCY_data/stringency.png', bbox_inches='tight')
    # granger(Xdates, Y, Y1, policyRec)
    return


def granger(Xdates, Y, Y1, policyRec):
    policies = pd.Series()
    for policy in policyRec[1:]:
        policies = pd.concat([policies, policy])
    Y.index = Xdates
    Y1.index = Xdates
    Y = Y[Y.index >= policies.index[0]]
    Y1 = Y1[Y1.index >= policies.index[0]]
    Y.name = 'Y'
    Y1.name = 'Y1'
    policies.name = 'peak'
    df = pd.merge(Y, policies, right_index=True, left_index=True)
    print('******************************')
    print('Unvaccinated:')
    print('******************************')
    grangercausalitytests(df, maxlag=30)

    df = pd.merge(Y1, policies, right_index=True, left_index=True)
    print('******************************')
    print('Vaccinated:')
    print('******************************')
    grangercausalitytests(df, maxlag=30)
    return


def main():
    confirmed, days = read_NS_cases('2/1/20', '8/1/20')
    read_oxford_data(confirmed, days)
    return


if __name__ == '__main__':
    main()
