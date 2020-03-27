import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from scipy.optimize import curve_fit
from scipy import stats
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys
import sympy as sym
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ttest_ind

"""IMPORT DATA"""
"""
MANIPULATE DATA TO ACHIEVE DESIRED FORMAT: 
RETURNS THE FOLLOWING DICTIONARY
COUNTRY --> DATE --> CASES, RECOVERIES, DEATHS
"""
def import_data():
    end_dictionary = {} #Output dictionary[country][date] --> cases, recoveries, deaths

    death = pd.read_csv("DATA/deaths_long.CSV")  #The lengths of these dataframes are all different
    death = death.drop(death.index[0])
    death['Value'] = pd.to_numeric(death['Value'])

    recover = pd.read_csv("DATA/recover_long.CSV")
    recover = recover.drop(recover.index[0])
    recover['Value'] = pd.to_numeric(recover['Value'])

    case = pd.read_csv("DATA/confirmed_long.CSV")
    case = case.drop(case.index[0])
    case['Value'] = pd.to_numeric(case['Value'])

    unique_countries = set(death['Country/Region'])
    unique_dates = set(recover['Date'])

    for country in unique_countries:
        date = {}
        for d in unique_dates:
            total_deaths = np.sum(death[(death['Country/Region']==country) & (death['Date'] == d)]['Value'])
            total_recoveries = np.sum(recover[(recover['Country/Region']==country) & (recover['Date'] == d)]['Value'])
            total_cases = np.sum(case[(case['Country/Region'] == country) & (case['Date'] == d)]['Value'])
            #total_deaths = np.sum(death.loc[(death['Country/Region'] == country) & (death['Date'] == d)]['Value'])
            #total_recoveries = np.sum(recover.loc[(recover['Country/Region'] == country) & (recover['Date'] == d)]['Value'])
            #total_cases = np.sum(case.loc[(case['Country/Region'] == country) & (case['Date'] == d)]['Value']) #Locations where country =us, date = 3/26
            date[d] = np.array([int(total_cases), int(total_recoveries), int(total_deaths)])

        end_dictionary[country] = date.copy()
    return end_dictionary

"""RELATE DATA"""
"""
TAKES INPUT DICTIONARY FROM IMPORT_DATA FUNCTION
RETURNS A DICTIONARY WHERE CASES, RECOVERIES, DEATHS ARE PER DAY and SUMMED IN TOTAL
DICTIONARY[COUNTRY][DATE]--> RETURNS [[CASES, rECOVERIS, DEATHS], [CASES, RECOVERIES, DEATHS]] (SUMMED, BY DAY) 
"""
def add_on(dictionary):
    new_dictionary = copy.deepcopy(dictionary)
    output_dictionary = copy.deepcopy(dictionary)
    dates = list(new_dictionary['US'].keys())
    dates.sort(key=lambda date: datetime.strptime(date, "%m/%d/%y"))
    dates.reverse()
    for country in new_dictionary.keys():
        for a in range(0, len(dates)-1):
            current_data = np.subtract(new_dictionary[country][dates[a]], new_dictionary[country][dates[a+1]])
            output_dictionary[country][dates[a]] = [new_dictionary[country][dates[a]], current_data]
        output_dictionary[country][dates[len(dates)-1]] = [new_dictionary[country][dates[len(dates)-1]], [0, 0, 0]]
    return output_dictionary

def by_nth_case(complete_dictionary, n):
    helper = copy.deepcopy(complete_dictionary)
    return_dict = {}
    for country in helper.keys():
        country_dict = {}
        global d
        d = date_array[0]
        while helper[country][d][0][0] < n:
            new_index = date_array.index(d)+1
            if new_index == len(date_array):
                break
            d = date_array[date_array.index(d)+1]
        start_index = date_array.index(d)
        for a in range(start_index, len(date_array)):
            country_dict[a-start_index+1] = helper[country][date_array[a]]
        return_dict[country] = country_dict
    return return_dict

"""Takes in an array of countries (can be a single entry)
Plots a graph of total cases over time by date"""
def plot_total_cases_datewise(countries):
    array_per_country = []
    for c in countries:
        case_array = []
        for a in date_array:
            case_array.append(complete_dictionary[c][a][0][0])
        array_per_country.append(case_array)

    for a in range(0, len(countries)):
        plt.scatter(date_array, array_per_country[a], label = countries[a], color=colors_2[a])

    plt.xlabel("Date")
    plt.ylabel("Total Number of Cases")
    plt.title("Total COVID-19 Cases for Different Countries")
    plt.legend()
    locs, labels = plt.xticks()
    plt.xticks(locs[0::10])

"""Takes in an array of countries
Plots a graph of total cases over time since the first case was identified"""
def plot_total_cases_offset(countries):
    array_per_country = []
    for c in countries:
        case_array = []
        for a in range(1, len(offset_dictionary[c].keys())+1):
            case_array.append(offset_dictionary[c][a][0][0])
        array_per_country.append(case_array)

    for a in range(0, len(countries)):
        if countries[a]=='China':
            plt.scatter(range(68, 68+len(array_per_country[a])), array_per_country[a], label=countries[a], c=colors_2[a])
        else:
            plt.scatter(range(0, len(array_per_country[a])), array_per_country[a], label=countries[a], c = colors_2[a])

    plt.xlabel("Days Since First Confirmed Case")
    plt.ylabel("Total Number of Cases")
    plt.title("Total COVID-19 Cases for Different Countries")
    plt.legend()

def c_r_d(countries):
    current_date = date_array[-1]
    c_array = []
    r_array = []
    d_array = []
    for c in countries:
        data = complete_dictionary[c][current_date]
        c_array.append(data[0][0])
        r_array.append(data[0][1])
        d_array.append(data[0][2])
    fig, ax = plt.subplots()
    index = np.arange(len(countries))
    bar_width = 0.3
    opacity = 0.8

    rects1 = plt.bar(index, c_array, bar_width, alpha = opacity, color = 'b', label = "Total Cases")
    rects2 = plt.bar(index + bar_width, r_array, bar_width, alpha = opacity, color = 'orange', label = 'Total Recoveries')
    rects3 = plt.bar(index + bar_width + bar_width, d_array, bar_width, alpha = opacity, color = 'r', label = 'Total Deaths')

    plt.xlabel("Country")
    plt.ylabel("Count")
    plt.title("Cases, Recoveries, and Deaths per Country")
    plt.xticks(index + bar_width, countries)
    plt.legend()
    plt.tight_layout()
    plt.show()

def statistics_overtime(countries):
    c_counts = []
    c_final = {}
    r_final = {}
    d_final = {}
    for c in countries:
        num = range(1, len(offset_dictionary[c]))
        num_eff = num
        if c=='China':
            num = range(1+68, 68+len(offset_dictionary[c]))
        else:
            num = range(1, len(offset_dictionary[c]))
        c_counts.append(num)
        c_array = []
        r_array = []
        d_array = []
        for val in num_eff:
            data = offset_dictionary[c][val][0]
            c_array.append(data[0])
            r_array.append(data[1])
            d_array.append(data[2])
        c_final[countries.index(c)] = c_array
        r_final[countries.index(c)] = np.divide(r_array, c_array)*100
        d_final[countries.index(c)] = np.divide(d_array, c_array)*100

    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True) #3 rows, 2 columns
    fig.suptitle('Cases, Recovery Rate, and Death Rate for Multiple Countries')
    global x, y
    x = 0
    y = 0
    total = 0
    while x < 3 or y < 1:
        if x==3:
            break
        ax[x, y].plot(c_counts[total], c_final[total], color = 'r')
        ax[x, y].set_xlim([0, 150])
        #ax[x, y].set_ylim([0, 50000])
        ax_additional = ax[x, y].twinx()
        ax_additional.plot(c_counts[total], r_final[total], '--', color = 'b', label = 'Recovery Rate')
        ax_additional.plot(c_counts[total], d_final[total], ':', color = 'b', label = 'Death Rate')
        ax_additional.set_ylim([0, 100])
        ax_additional.tick_params(axis='y', labelcolor='b')
        ax[x, y].set_title(countries[total])
        if y ==0:
            ax_additional.set_yticklabels([])
            y+=1
        elif y==1:
            y=0
            x+=1
        total+=1
    for a in ax.flat:
        a.label_outer()
    ax[1, 1].yaxis.set_label_position("right")
    ax[1, 1].yaxis.set_label_coords(1.2, .5)
    ax[1, 1].set_ylabel("Percent (%)", color = 'b')
    fig.text(0.5, 0.04, 'Days Since First Confirmed Case', ha='center')
    fig.text(0.04, 0.5, 'Case Count', va='center', rotation='vertical')
    handles, labels = ax_additional.get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.show()
#returns growth rates for the first X days of the disease for specified countries
def exponential_growths(countries, n, bool):
    return_dict = {}
    for c in countries:
        if c == 'China' or c=='Korea, South':
            n_copy = int(n/2)
        else:
            n_copy = n
        max_len = min(len(tenth_dictionary[c]), n_copy)
        #if c=='China':
           # max_len = 15
        xdata = range(1, max_len+1)
        ydata = [tenth_dictionary[c][i][0][0] for i in xdata]
        popt, pcov = curve_fit(exponential, xdata, ydata, method='dogbox')
        return_dict[c] = popt
        x = range(1, len(tenth_dictionary[c]))
        y = exponential(xdata, *popt)
        r_sq = r_squared(xdata, ydata, y)
        if bool:
            plt.plot(xdata, y, color = colors[c])
            plt.scatter(x, [tenth_dictionary[c][i][0][0] for i in x], label = (c + ': R-Squared = ' + str(round(r_sq, 3))), color = colors[c], alpha=0.75)
            plt.xlabel("Days Since First Confirmed Case")
            plt.ylabel("Total Cases")
            plt.title("Fitting The Initial Growth of COVID-19 Cases Exponentially")
    if bool:
        plt.legend(loc='upper left')
        plt.show()
    else:
        return return_dict
"""
Break Data into two chunks: range of data near beginning, range near end
do linear regression: Train on randomly selected 80% of those values, test on 20%
italy: 0:15, 16:31
US: 0:15, 16:31
south korea: 11:23 24:36
china: 0:15 16:31
germany: 0:14 15:30
iran: 0:15 15:30
Takes in an input dictionary of ranges
"""
def linear_fits(countries, xdata_c, bool): #Goal: compute 2 linear fits on different portions of the data; compare their standard errors; xdat should be a range
    return_dict = {}
    plots_countries = {}
    for c in countries:
        strt_dict = {}
        for strt in range(0, 2):
            # if c=='China':
            # max_len = 15
            xdata = xdata_c[c][strt]
            ydata = np.log([tenth_dictionary[c][i][0][0] for i in xdata])
            m, b, r_value, p_value, std_err = stats.linregress(xdata, ydata)
            strt_dict[strt] = [m, b, r_value, p_value, std_err]
            x = range(1, len(tenth_dictionary[c])+1)
            y = m*xdata+b
            #r_sq = r_squared(xdata, ydata, y)
            if bool:
                plt.plot(xdata, y, color=colors[c], alpha = 0.95)
                if strt ==1:
                    plots_countries[c] = plt.scatter(x, np.log([tenth_dictionary[c][i][0][0] for i in x]),
                                color=colors[c], alpha=0.65)
                plt.xlabel("Days Since First Confirmed Case")
                plt.ylabel("Log Total Cases")
                plt.title("Fitting The Initial Growth of COVID-19 Cases Exponentially")
        return_dict[c] = strt_dict

    if bool:
        legend_plots = []
        legend_labels = []
        for c in countries:
            legend_plots = np.append(legend_plots, plots_countries[c])
            st = c + ' R-Squared: ' + str(round(return_dict[c][0][2], 3)) + '; ' + str(round(return_dict[c][1][2], 3))
            legend_labels = np.append(legend_labels, st)
        plt.legend(legend_plots, legend_labels, loc='lower right') #label=(c + ': R-Squared = ' + str(round(r_sq, 3))),
        plt.show()
    else:
        return return_dict

"""Double bar graph per country of slope values with double standard error bars"""
"""Takes in output of linear_fits: a dictionary[country][0 or 1] = m, b, p_value, r_value, std_err"""
def linear_evaluation(countries, return_dict):
    a_m = []
    a_error = []
    b_m = []
    b_error = []
    for c in countries:
        data = return_dict[c]
        a_m.append(data[0][0])
        a_error.append(data[0][4])

        b_m.append(data[1][0])
        b_error.append(data[1][4])

    country_p = []
    for a in range(0, len(countries)):
        n = countries_n[a]
        p_value = ttest_ind_from_stats(a_m[a], b_m[a], n, a_error[a]*(n**.5), b_error[a]*(n**.5), n)[1]
        country_p.append(p_value)

    fig, ax = plt.subplots()
    index = np.arange(len(countries))
    bar_width = 0.3
    opacity = 0.8

    rects1 = plt.bar(index, a_m, bar_width, yerr = tuple(a_error), alpha=opacity, color='b', label="Initial Fit")
    rects2 = plt.bar(index + bar_width, b_m, bar_width, yerr = tuple(b_error), alpha=opacity, color='orange', label='Later Fit')
    font = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 16}
    for a in range(0, len(rects1)):
        height = max(rects1[a].get_height(), rects2[a].get_height()) + 0.008
        x = int((rects1[a].get_x() + rects2[a].get_x())/2)
        ast = '*'
        if p_value < 0.01:
            ast = '**'
        elif p_value < 0.001:
            ast = '***'
        p_value = country_p[a]
        plt.text(x,height, '     ' + ast, fontdict=font, ha ='center', va='bottom')



    plt.xlabel("Country")
    plt.ylabel("Log Linear Slope")
    plt.title("Comparing Log Linear Fit of Cases per Country")
    plt.xticks(index + bar_width, countries)
    plt.legend()
    plt.tight_layout()
    plt.show()

def r_squared(xdata, ydata, y_fit):
    ss_res = np.sum((ydata-y_fit)**2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    return 1-(ss_res/ss_tot)

def exponential(x, b, initial_value):
    return initial_value*(b ** x)

def linear(x, a, b):
    return a*x + b

#countries_stats = ['Italy', 'US', 'Korea, South', 'Iran', 'China', 'Germany', 'Spain', 'France', 'United Kingdom', 'Switzerland', 'Netherlands', 'Austria']
def health_score(countries):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    h_s = {'China': 144, 'Germany': 25, 'US': 37, 'Iran': 93, 'Korea, South': 58, 'Italy': 2, 'Spain': 7, 'France': 1, 'United Kingdom': 18, 'Switzerland': 20, 'Netherlands': 17, 'Austria': 9}
    exp = linear_fits(countries, xdata_expanded, False)
    h_array = []
    e_one = []
    e_two = []
    for c in countries:
        axs[0].scatter(h_s[c], exp[c][0][0], label = c, color = colors[c])
        axs[1].scatter(h_s[c], exp[c][1][0], color = colors[c])
        h_array.append(h_s[c])
        e_one.append(exp[c][0][0])
        e_two.append(exp[c][1][0])
    pearson_coefficient_one, p_value_one = stats.pearsonr(h_array, e_one)
    pearson_coefficient_two, p_value_two = stats.pearsonr(h_array, e_two)
    pearson = [pearson_coefficient_one, pearson_coefficient_two]
    p = [p_value_one, p_value_two]

    xlim = .4*plt.gca().get_xlim()[1]
    ylim = [0.1, 0.4]
    for a in range(0, 2):
        x_box_start = xlim + .2
        y_box_start = ylim[a]
        axs[a].text(x_box_start, y_box_start, 'R: ' + str(round(pearson[a], 2)) + ', P-Value: ' + str(round(p[a], 3)), style='italic',
                 bbox={'facecolor': 'red', 'alpha': 0.7, 'pad': 7})

    fig.text(0.5, 0.004, "Health Care Rating", ha = 'center')
    axs[0].set_ylabel("Initial Log Linear Growth Rate")
    axs[1].set_ylabel("Later Log Linear Growth Rate")
    fig.suptitle("Comparing Health Care Quality and COVID-19 Growth")
    axs[0].legend()

def democracy_score(countries):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    h_s = {'China': 2.26, 'Germany': 8.68, 'US': 7.96, 'Iran': 2.38, 'Korea, South': 8, 'Italy': 7.52, 'Spain': 8.29, 'France': 8.12,
           'United Kingdom': 8.52, 'Switzerland': 9.03, 'Netherlands': 9.01, 'Austria': 8.29}
    exp = linear_fits(countries, xdata_expanded, False)
    h_array = []
    e_one = []
    e_two = []
    for c in countries:
        axs[0].scatter(h_s[c], exp[c][0][0], label=c, color=colors[c])
        axs[1].scatter(h_s[c], exp[c][1][0], color=colors[c])
        h_array.append(h_s[c])
        e_one.append(exp[c][0][0])
        e_two.append(exp[c][1][0])
    pearson_coefficient_one, p_value_one = stats.pearsonr(h_array, e_one)
    pearson_coefficient_two, p_value_two = stats.pearsonr(h_array, e_two)
    pearson = [pearson_coefficient_one, pearson_coefficient_two]
    p = [p_value_one, p_value_two]

    xlim = .5 * plt.gca().get_xlim()[1]
    ylim = [0.08, 0.4]
    for a in range(0, 2):
        x_box_start = xlim + .25
        y_box_start = ylim[a]
        axs[a].text(x_box_start, y_box_start, 'R: ' + str(round(pearson[a], 2)) + ', P-Value: ' + str(round(p[a], 3)),
                    style='italic',
                    bbox={'facecolor': 'red', 'alpha': 0.7, 'pad': 7})

    fig.text(0.5, 0.004, "Democracy Index", ha='center')
    axs[0].set_ylabel("Initial Log Linear Growth Rate")
    axs[1].set_ylabel("Later Log Linear Growth Rate")
    fig.suptitle("Comparing Democracy Indices and COVID-19 Growth")
    axs[0].legend()

def population_score(countries):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    h_s = {'China': 377, 'Germany': 603, 'US': 87, 'Iran': 131, 'Korea, South': 1339, 'Italy': 518, 'Spain': 241,
           'France': 7.3,
           'United Kingdom': 710, 'Switzerland': 539, 'Netherlands': 1088, 'Austria': 275}
    exp = linear_fits(countries, xdata_expanded, False)
    h_array = []
    e_one = []
    e_two = []
    for c in countries:
        axs[0].scatter(h_s[c], exp[c][0][0], label=c, color=colors[c])
        axs[1].scatter(h_s[c], exp[c][1][0], color=colors[c])
        h_array.append(h_s[c])
        e_one.append(exp[c][0][0])
        e_two.append(exp[c][1][0])
    pearson_coefficient_one, p_value_one = stats.pearsonr(h_array, e_one)
    pearson_coefficient_two, p_value_two = stats.pearsonr(h_array, e_two)
    pearson = [pearson_coefficient_one, pearson_coefficient_two]
    p = [p_value_one, p_value_two]

    xlim = .4 * plt.gca().get_xlim()[1]
    ylim = [0.1, 0.3]
    for a in range(0, 2):
        x_box_start = xlim + .2
        y_box_start = ylim[a]
        axs[a].text(x_box_start, y_box_start, 'R: ' + str(round(pearson[a], 2)) + ', P-Value: ' + str(round(p[a], 3)),
                    style='italic',
                    bbox={'facecolor': 'red', 'alpha': 0.7, 'pad': 7})

    fig.text(0.5, 0.004, "Population Density (pop/mi^2)", ha='center')
    axs[0].set_ylabel("Initial Log Linear Growth Rate")
    axs[1].set_ylabel("Later Log Linear Growth Rate")
    fig.suptitle("Comparing Population Density and COVID-19 Growth")
    axs[0].legend()

#Format: country = [lockdown, testing, hospital, medical supply surge, school closures, location tracking], where each value is days after first case.
def response_times(countries):
    ind = np.arange(len(countries))
    width = 0.1
    lthmsl = [[], [], [], [], [], []]
    for a in range(0, len(US)):
        lthmsl[a] = [US[a], italy[a], china[a], south_korea[a]]

    p1 = plt.bar(ind, lthmsl[0], width, label = policy[0], color=policy_colors[0])
    p2 = plt.bar(ind+ width, lthmsl[1], width, label = policy[1], color=policy_colors[1])
    p3 = plt.bar(ind+2*width, lthmsl[2], width, label = policy[2], color=policy_colors[2])
    p4 = plt.bar(ind+3*width, lthmsl[3], width, label = policy[3], color=policy_colors[3])
    p5 = plt.bar(ind+4*width, lthmsl[4], width, label = policy[4], color=policy_colors[4])
    p6 = plt.bar(ind+5*width, lthmsl[5], width, label = policy[5], color=policy_colors[5])
    plt.legend()
    plt.xlabel("Country")
    plt.ylabel("Days after First Confirmed Case")
    plt.title("Policy Response Delay to COVID-19")
    plt.xticks(ind + width+0.12, countries)
    plt.tight_layout()

def policy_plot(countries):
    val_subset = [US, italy, china, south_korea]
    country_names = 'US', 'Italy', 'China', "Korea, South"
    c_counts = []
    c_final = {}
    for c in countries:
        num = range(1, len(offset_dictionary[c]))
        num_eff = num
        if c == 'China':
            num = range(1 + 68, 68 + len(offset_dictionary[c]))
        else:
            num = range(1, len(offset_dictionary[c]))
        c_counts.append(num)
        c_array = []
        for val in num_eff:
            data = offset_dictionary[c][val][0]
            c_array.append(data[0])
        c_final[countries.index(c)] = c_array

    fig, axs, = plt.subplots(2, 2, sharex=True, sharey=True)
    ax = axs.flat
    for a in range(0, len(ax)):
        ax[a].set_title(country_names[a])
        ax[a].plot(c_counts[a], c_final[a])
        for b in range(0, len(US)):
            if val_subset[a][b] >0:
                ax[a].axvline(val_subset[a][b], ymin=0, ymax=1, color=policy_colors[b], label = policy[b])
    ax[1].legend()
    fig.suptitle("Policy Responses by Country")
    fig.text(0.5, 0.004, "Days Since First Confirmed Case", ha='center')
    fig.text(0.004, 0.5, "Total Cases (Count)", va='center', rotation = 'vertical')

"""Calculates a log linear model for the 10 days before the first day of a policy, and then incremenet after;"""
def policy_efficacy(index):
    C = china-68
    I = italy+6
    S = south_korea-2
    N = 20
    c_values = ['China', 'Italy', 'Korea, South']
    index_values = [C[index], I[index], S[index]]

    lin_reg = {}
    for a in range(0, len(c_values)):
        i = index_values[a]
        xdata = range(i-11, i)
        ydata = []
        for b in xdata:
            ydata.append(offset_dictionary[c_values[a]][b][0][0])
        m, b, r_value, p_value, std_err = stats.linregress(xdata, np.log(ydata))
        lin_reg[c_values[a]] = [m, b, r_value, p_value, std_err]

    days = {}
    for a in range(0, len(c_values)):
        c = c_values[a]
        computed_p = 1
        n_days = 0
        while computed_p > 0.001:
            n_days += 1
            i = index_values[a] + n_days
            xdata = range(i - 11, i)
            ydata = []
            for b in xdata:
                ydata.append(offset_dictionary[c_values[a]][b][0][0])
            m, b, r_value, p_value, std_err = stats.linregress(xdata, np.log(ydata))
            computed_p = ttest_ind_from_stats(lin_reg[c][0], m, N, lin_reg[c][4] * (N ** .5), std_err * (N ** .5), N)[1]
        days[c] = n_days
    return days

def policy_efficacy_graph():
    medical_supplies = policy_efficacy(3)
    school_closure = policy_efficacy(4)

    mean_med = np.mean(list(medical_supplies.values()))
    std_med = np.std(list(medical_supplies.values()))/np.sqrt(3)

    mean_school = np.mean(list(school_closure.values()))
    std_school = np.std(list(school_closure.values()))/np.sqrt(3)

    index = np.arange(1)
    bar_width = 0.3
    opacity = 0.8

    rects1 = plt.bar(index, mean_med, bar_width, yerr=std_med, alpha=opacity, color=policy_colors[3], label="Medical Supply Acquisition")
    rects2 = plt.bar(index+bar_width, mean_school, bar_width, yerr=std_school, alpha=opacity, color=policy_colors[4], label="School Closure")

    plt.xlabel("Policy")
    plt.legend()
    plt.ylabel("Mean Days Until Significant Log Linear Growth Change")
    plt.title("Analyzing Log Linear Growth Change with Respect to Policy Onset Dates")
    font = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 16}
    height = max(rects1[0].get_height(), rects2[0].get_height()) +0.35
    x = int((rects1[0].get_x() + rects2[0].get_x()) / 2)
    p_value = ttest_ind(list(school_closure.values()), list(school_closure.values()))[1]
    plt.text(x, height, '                                    P-Value: ' + str(round(p_value, 3)), fontdict=font, ha='center', va='bottom')



def carrying_capacity(countries, d):
    #dt=50
    exp = exponential_growths(countries, d, False)
    k_dictionary = {}
    for c in countries:
        #dt = 50
        dt = d
        if c == 'China' or c == 'Korea, South':
            dt = dt/2
        true_n = min(len(tenth_dictionary[c]), dt)
        x = sym.Symbol('x')
        growth_rate, constant = exp[c]
        N = tenth_dictionary[c][true_n][0][0]
        dn_dt = sym.diff(constant*(growth_rate**x)).subs(x, true_n)
        day = 15
        dn_dt = tenth_dictionary[c][day][0][0]
        #N = constant*growth_rate**day
        k_value = calculate_K(growth_rate, N, dn_dt)
        k_dictionary[c] = k_value
    return k_dictionary

def exp_sig_fit(countries, d):
    k_values = carrying_capacity(countries, d)
    exp_values = exponential_growths(countries, d, False)
    x_range = range(1, len(tenth_dictionary['China'])+1)
    d_range = range(1, int(d/2))
    y_exp = exponential(d_range, exp_values['China'][0], exp_values['China'][1])
    y_sig = sigmoid_simple(x_range, k_values['China'], exp_values['China'][0])
    plt.scatter(x_range, [tenth_dictionary['China'][i][0][0] for i in x_range])
    plt.plot(d_range, y_exp, label = 'Exponential Fit')
    plt.plot(x_range, y_sig, label = 'Sigmoidal Fit')

def calculate_K(r, N, dn_dt):
    return N/(1-dn_dt/(N*r))

def sigmoid(x, L, x0, k, b):
    return (L / (1 + np.exp(-k*(x-x0)))+b)

def sigmoid_simple(x, L, r):
    return (L / (1 + r**x))

observed_K = {}
for a in range(10, 60):
    observed_K[a] = carrying_capacity(countries, a)['China']


##LOAD IN/CREATE THE DATA
summed_data = import_data()
complete_dictionary = add_on(summed_data) #Dictionary by literal dates
pickle.dump( complete_dictionary, open( "DATA/covid19_3-26.p", "wb" ) )
complete_dictionary = pickle.load(open("DATA/covid19_3-26.p", 'rb'))
global date_array;
date_array = list(complete_dictionary['US'].keys())
date_array.sort(key=lambda date: datetime.strptime(date, "%m/%d/%y"))
offset_dictionary = by_nth_case(complete_dictionary, 1) #Dictionary by day of first case: index 1:end
tenth_dictionary = by_nth_case(complete_dictionary, 25) #Reorganize in terms of how many days before 1000 cases??

#Goals: Need the dates by days since first case.  DONE

#A plot with the cases over time for multiple countries
mpl.style.use('seaborn')
#mpl.style.use('default')
colors = {'Italy': 'red', 'US': 'blue', 'Korea, South': 'green', 'Iran': 'orange', 'China': 'brown', 'Germany': 'purple', 'Spain': 'paleturquoise', 'France': 'orchid', 'United Kingdom': 'deeppink', 'Switzerland': 'sienna', 'Netherlands': 'lightslategray', 'Austria': 'palegreen'}
colors_2 = ['red', 'blue', 'green', 'orange', 'brown', 'purple', 'paleturquoise', 'orchid', 'deeppink', 'sienna', 'lightslategray', 'palegreen', 'lightsalmon']
countries = ['Italy', 'US', 'Korea, South', 'Iran', 'China', 'Germany']
plot_total_cases_datewise(countries)
plot_total_cases_offset(countries)
#Subplots with deaths, cases, and recoveries for many countries
statistics_overtime(countries)
#A triple bar graph of deaths, cases, and recoveries for multiple countries.
c_r_d(countries)
#A plot with vertical bars at the time at which policy changes were made
#A plot of health care rank vs growth of cases  MORE COUNTRIES; INITIAL LINEAR FITS
countries_stats = ['Italy', 'US', 'Korea, South', 'Iran', 'China', 'Germany', 'Spain', 'France', 'United Kingdom', 'Switzerland', 'Netherlands', 'Austria']
xdata_expanded = {'Italy': [range(1, 15), range(16, 30)], 'US': [range(5, 17), range(18, 30)], 'Korea, South': [range(11, 23), range(24, 36)], 'China': [range(1, 17), range(18, 34)], 'Germany': [range(1, 14), range(15, 29)], 'Iran': [range(1, 15), range(16, 30)], 'Spain': [range(1, 12), range(13, 24)], 'France': [range(1, 12), range(13, 15)], 'United Kingdom': [range(1, 13), range(14, 25)], 'Switzerland': [range(1, 10), range(11, 20)], 'Netherlands': [range(1, 10), range(11, 20)], 'Austria': [range(1, 12), range(13, 24)]}
health_score(countries_stats)
#A plot of authoritarian rank vs growth of cases MORE COUNTRIES; INITIAL LINEAR FITS
democracy_score(countries_stats)
#A plot of population density and growth of cases MORE COUNTRIES; INITIAL LINEAR FITS
population_score(countries_stats)
#Fit the initial part of the graph exponentially; then plot the sigmoidal; r^2 value
exponential_growths(countries, 40, True)
#Linear fits on log cases
xdata_vibes = {'Italy': [range(1, 15), range(16, 30)], 'US': [range(5, 17), range(18, 30)], 'Korea, South': [range(11, 23), range(24, 36)], 'China': [range(1, 17), range(18, 34)], 'Germany': [range(1, 14), range(15, 29)], 'Iran': [range(1, 15), range(16, 30)]}
linear_fits(countries, xdata_vibes, True)
linear_input = linear_fits(countries, xdata_vibes, False)
countries_n = [13, 11, 11, 15, 12, 13]
linear_evaluation(countries, linear_input)

#Plots for each country with policy bars on them
policy_plot(countries_subset) #Need the below data to do this

#Plots comparing response times
response_times(countries_subset) #Need the below data to do this

#Calculate policy effectiveness
policy_efficacy(3) #input 3 for medical supplies; 4 for school closure; calculates at p-value = 0.001

#Plot Policy effectiveness
policy_efficacy_graph()

#Format: country = [lockdown, testing, hospital, medical supply surge, school closures, location tracking], where each value is days after first case.
#Index: cases per capita / initial growth rate
countries_subset = ['US', 'Italy', 'China', 'Korea, South']
policy = ['Residential Lockdown', 'Ramped Testing', 'Hospital Construction', 'Medical Supply Acquisition', 'School Closure', 'Location Tracking']
policy_colors = ['r', 'b', 'g', 'deeppink', 'paleturquoise', 'orchid']
china = np.array([68, 0, 78, 84, 92, 85]) #China must be treated specially since first case was so long ago.plot = actual - 66
south_korea = np.array([32, 30, 0, 37, 42, 17])  #Offset is 2 days: actual -2
italy = np.array([38, 53, 35, 35, 34, 0]) #Offset is 6 days: actual + 6; over 300k tests done by that date
US = np.array([0, 45, 0, 55, 46, 0]) #Offset is 6 days: actual + 6
#0: lockdown #1: available testing
korea_start = '2020-01-20'
china_start = '2020-11-17'
italy_start = '2020-01-30'
US_start = '2020-01-30'

















us = complete_dictionary['US']
death_total = []
death_running = []
date = sorted(date)
for a in date[1:len(date)]:
    death_total = np.append(death_total, us[a][0][0])
    death_running = np.append(death_running, us[a][1][0])
death_total_us = death_total
plt.scatter(date[1:len(date)], death_total, label = "D")
plt.scatter(date[1:len(date)], death_total_us, label = "USA")
plt.legend()
plt.scatter(date[1:len(date)], death_running, color = 'red')
initial_value = complete_dictionary['Korea, South']['2020-01-22'][0]
initial_value = 433

def sigmoid(x, L, x0, k, b):
    return (L / (1 + np.exp(-k*(x-x0)))+b)

xrange = len(death_total)
xrange = 12
offset = 30
xdata = [*range(offset, offset+xrange, 1)]
ydata = death_total[offset:xrange+offset]
p0 = [100000, np.median(xdata), 1, min(ydata)]
popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox', maxfev=20000)
x = np.array(list(range(0, 50)))
y = sigmoid(x, *popt)
plt.plot(x, y, label = 'Fit')
plt.scatter(date[1:len(date)], death_total, color = 'orange', label = 'Data')


popt2, pcov2 = curve_fit(exponential_sigmoid, xdata, ydata, method='dogbox')
x = np.array(list(range(0, 50)))
y = exponential_sigmoid(x, *popt2)
plt.plot(x, y, label = 'Fit')
plt.scatter(date[1:len(date)], death_total, color = 'orange', label = 'Data')


dt = 12+offset
#dn_dt = (44759-548)/dt
#N = 44759
#dn_dt = 20452.8824635
dn_dt = 1651.65
N = exponential_sigmoid(dt, *popt2)
r = popt2[0]



K = calculate_K(r, N, dn_dt)



x = range(0, 70) #FIGURE OUT B? R?
sig_y = sigmoid_custom(K, initial_value, r-1, x)
plt.plot(x, sig_y)
plt.scatter(date[1:len(date)], death_total, color = 'orange', label = 'Data')
