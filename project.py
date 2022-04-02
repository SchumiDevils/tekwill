from csv import unregister_dialect
from this import d
import matplotlib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
plt.style.use('fivethirtyeight')

confirmed_cases=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_reported=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_cases=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/07-15-2020.csv')

# Se crează coloane din baza de date confirmată
cols=confirmed_cases.keys()
# Se extrag coloanele de date
confirmed=confirmed_cases.loc[:, cols[4]:cols[-1]]
deaths=deaths_reported.loc[:, cols[4]:cols[-1]]
recoveries=recovered_cases.loc[:, cols[4]:cols[-1]]

# Interval de date
dates=confirmed.keys()

# Suma
world_cases=[]
total_deaths=[]
mortality_rate=[]
recovery_rate=[]
total_recovered=[]
total_active=[]

# Confirmați
china_cases=[]
italy_cases=[]
us_cases=[]
spain_cases=[]
france_cases=[]
germany_cases=[]
uk_cases=[]
russia_cases=[]
india_cases=[]

# Morți
china_deaths=[]
italy_deaths=[]
us_deaths=[]
spain_deaths=[]
france_deaths=[]
germany_deaths=[]
uk_deaths=[]
russia_deaths=[]
india_deaths=[]

# Recuperați
china_recoveries=[]
italy_recoveries=[]
us_recoveries=[]
spain_recoveries=[]
france_recoveries=[]
germany_recoveries=[]
uk_recoveries=[]
russia_recoveries=[]
india_recoveries=[]

# Completăm cu setul de date
for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=deaths[i].sum()
    recovered_sum=recoveries[i].sum()

    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum-death_sum-recovered_sum)
    
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)

    china_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='China'][i].sum())
    italy_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Italy'][i].sum())
    us_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='US'][i].sum())
    spain_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Spain'][i].sum())
    france_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='France'][i].sum())
    germany_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Germany'][i].sum())
    uk_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='United Kingdom'][i].sum())
    russia_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Russia'][i].sum())
    india_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='India'][i].sum())

    china_deaths.append(deaths_reported[deaths_reported['Country/Region']=='China'][i].sum())
    italy_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Italy'][i].sum())
    us_deaths.append(deaths_reported[deaths_reported['Country/Region']=='US'][i].sum())
    spain_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Spain'][i].sum())
    france_deaths.append(deaths_reported[deaths_reported['Country/Region']=='France'][i].sum())
    germany_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Germany'][i].sum())
    uk_deaths.append(deaths_reported[deaths_reported['Country/Region']=='United Kingdom'][i].sum())
    russia_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Russia'][i].sum())
    india_deaths.append(deaths_reported[deaths_reported['Country/Region']=='India'][i].sum())

    china_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='China'][i].sum())
    italy_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Italy'][i].sum())
    us_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='US'][i].sum())
    spain_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Spain'][i].sum())
    france_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='France'][i].sum())
    germany_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Germany'][i].sum())
    uk_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='United Kingdom'][i].sum())
    russia_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Russia'][i].sum())
    india_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='India'][i].sum())

def daily_increase(data):
    d=[]
    for i in range(len(data)):
        if i==0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d

# Cazuri confirmate
world_daily_increase=daily_increase(world_cases)
china_daily_increase=daily_increase(china_cases)
italy_daily_increase=daily_increase(italy_cases)
us_daily_increase=daily_increase(us_cases)
spain_daily_increase=daily_increase(us_cases)
frace_daily_increase=daily_increase(france_cases)
germany_daily_increase=daily_increase(germany_cases)
uk_daily_increase=daily_increase(uk_cases)
india_daily_increase=daily_increase(india_cases)

# Decese
world_daily_death=daily_increase(total_deaths)
china_daily_death=daily_increase(china_deaths)
italy_daily_death=daily_increase(italy_deaths)
us_daily_death=daily_increase(us_deaths)
spain_daily_death=daily_increase(spain_deaths)
france_daily_death=daily_increase(france_deaths)
germany_daily_death=daily_increase(germany_deaths)
uk_daily_death=daily_increase(uk_deaths)
india_daily_death=daily_increase(india_deaths)

# Recuperări
world_daily_recovery=daily_increase(total_recovered)
china_daily_recovery=daily_increase(china_recoveries)
italy_daily_recovery=daily_increase(italy_recoveries)
us_daily_recovery=daily_increase(us_recoveries)
spain_daily_recovery=daily_increase(spain_recoveries)
france_daily_recovery=daily_increase(france_recoveries)
germany_daily_recovery=daily_increase(germany_recoveries)
uk_daily_recovery=daily_increase(uk_recoveries)
india_daily_recovery=daily_increase(india_recoveries)

unique_countries=list(latest_data['Country_Region'].unique())

confirmed_by_country=[]
death_by_country=[]
active_by_country=[]
recovery_by_country=[]
mortality_rate_by_country=[]

no_cases=[]
for i in unique_countries:
    cases=latest_data[latest_data['Country_Region']==i]['Confirmed'].sum()
    if cases>0:
        confirmed_by_country.append(cases)
    else:
        no_cases.append(i)
for i in no_cases:
    unique_countries.remove(i)

# Sortăm țările după numărul de cazuri confirmate
unique_countries=[k for k, v in sorted(zip(unique_countries, confirmed_by_country), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    confirmed_by_country[i]=latest_data[latest_data['Country_Region']==unique_countries[i]]['Confirmed'].sum()
    death_by_country.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Deaths'].sum())
    recovery_by_country.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Recovered'].sum())
    active_by_country.append(confirmed_by_country[i]-death_by_country[i]-recovery_by_country[i])
    mortality_rate_by_country.append(death_by_country[i]/confirmed_by_country[i])

country_df=pd.DataFrame({'Numele țării': unique_countries, 'Numărul de cazuri confirmate': confirmed_by_country,
                         'Numărul de decese': death_by_country, 'Numărul de recuperați': recovery_by_country,
                         'Numărul de cazuri active': active_by_country,
                         'Rata de mortalitate': mortality_rate_by_country})
# Numărul de cazuri per țară/regiune

unique_provinces=list(latest_data['Province_State'].unique())

confirmed_by_province=[]
country_by_province=[]
death_by_province=[]
recovery_by_province=[]
mortality_rate_by_province=[]

no_cases=[]
for i in unique_provinces:
    cases=latest_data[latest_data['Province_State']==i]['Confirmed'].sum()
    if cases>0:
        confirmed_by_province.append(cases)
    else:
        no_cases.append(i)

# Eliminăm zonele fără cazuri confirmate
for i in no_cases:
    unique_provinces.remove(i)

unique_provinces=[k for k, v in sorted(zip(unique_provinces, confirmed_by_province), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_provinces)):
    confirmed_by_province[i]=latest_data[latest_data['Province_State']==unique_provinces[i]]['Confirmed'].sum()
    country_by_province.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Country_Region'].unique()[0])
    death_by_province.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Deaths'].sum())
    recovery_by_province.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Recovered'].sum())
    mortality_rate_by_province.append(death_by_province[i]/confirmed_by_province[i])

# Numărul de cazuri pe provincie/stat/oraș
province_df=pd.DataFrame({'Numele Orașului/Provinciei': unique_provinces, 'Țara': country_by_province, 'Numărul de cazuri confirmate': confirmed_by_province,
                          'Numărul de decese': death_by_province, 'Numărul de recuperări': recovery_by_province,
                          'Rata de mortalitate': mortality_rate_by_province})
# Numărul de cazuri per țară/regiune

# Tratăm valorile lipsă
nan_indices=[]

# Obținem nan dacă există, de obicei este un float: float('nan')
for i in range(len(unique_provinces)):
    if type(unique_provinces[i])==float:
        nan_indices.append(i)

unique_provinces=list(unique_provinces)
confirmed_by_province=list(confirmed_by_province)

for i in nan_indices:
    unique_provinces.pop(i)
    confirmed_by_province.pop(i)

USA_confirmed=latest_data[latest_data['Country_Region']=='US']['Confirmed'].sum()
outside_USA_confirmed=np.sum(confirmed_by_country)-USA_confirmed
plt.figure(figsize=(16, 9))
plt.barh('USA', USA_confirmed)
plt.barh('În exteriorul USA', outside_USA_confirmed)
plt.title('Numărul de cazuri confirmate Coronavirus', size=20)
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

# Afișăm doar 10 țări cu cele mai multe cazuri confirmate, restul sunt grupate în cealaltă categorie
visual_unique_countries=[]
visual_confirmed_cases=[]
others=np.sum(confirmed_by_country[10:])

for i in range(len(confirmed_by_country[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(confirmed_by_country[i])

visual_unique_countries.append('Altele')
visual_confirmed_cases.append(others)

def plot_bar_graphs(x, y, title):
    plt.figure(figsize=(16, 9))
    plt.barh(x, y)
    plt.title(title, size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

plot_bar_graphs(visual_unique_countries, visual_confirmed_cases, 'Numărul de cazuri confirmate Covid-19 în Țări/Regiuni')

def plot_pie_charts(x, y, title):
    c=random.choices(list(mcolors.CSS4_COLORS.values()),k=len(unique_countries))
    plt.figure(figsize=(12,12))
    plt.title(title, size=20)
    plt.pie(y, colors=c)
    plt.legend(x, loc='best', fontsize=15)
    plt.show()

plot_pie_charts(visual_unique_countries, visual_confirmed_cases, 'Cazuri confirmate Covid-19 per Țări')

# Afișăm doar 10 provincii cu cele mai multe cazuri confirmate, restul sunt grupate în cealaltă categorie
visual_unique_provinces=[]
visual_confirmed_cases2=[]
others=np.sum(confirmed_by_province[10:])

for i in range(len(confirmed_by_province[:10])):
    visual_unique_provinces.append(unique_provinces[i])
    visual_confirmed_cases2.append(confirmed_by_province[i])

visual_unique_provinces.append('Altele')
visual_confirmed_cases2.append(others)

plot_bar_graphs(visual_unique_provinces, visual_confirmed_cases2, 'Numărul de cazuri confirmate Coronavirus în Orașe/Provincii')

def plot_pie_country_with_regions(country_name, title):
    regions=list(latest_data[latest_data['Country_Region']==country_name]['Province_State'].unique())
    confirmed_cases=[]
    no_cases=[]

    for i in regions:
        cases=latest_data[latest_data['Province_State']==i]['Confirmed'].sum()
        if cases>0:
            confirmed_cases.append(cases)
        else:
            no_cases.append(i)

    # Eliminăm zonele fără cazuri confirmate
    for i in no_cases:
        regions.remove(i)

    # Arătăm doar primele 10 state
    regions=[k for k, v in sorted(zip(regions, confirmed_cases), key=operator.itemgetter(1), reverse=True)]

    for i in range(len(regions)):
        confirmed_cases[i]=latest_data[latest_data['Province_State']==regions[i]]['Confirmed'].sum()

    # Oraș/provincie suplimentară va fi considerată „altele”
    if(len(regions)>10):
        regions_10=regions[:10]
        regions_10.append('Altele')
        confirmed_cases_10=confirmed_cases[:10]
        confirmed_cases_10.append(np.sum(confirmed_cases[10:]))
        plot_pie_charts(regions_10, confirmed_cases_10, title)
    else:
        plot_pie_charts(regions, confirmed_cases, title)

plot_pie_country_with_regions('US', 'Cazurile confirmate COVID-19 în Statele Unite ale Americii')

plot_pie_country_with_regions('France', 'Cazurile confirmate COVID-19 în Franța')

# Prezicerea viitorului

days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases=np.array(world_cases).reshape(-1, 1)
total_deaths=np.array(total_deaths).reshape(-1, 1)
total_recovered=np.array(total_recovered).reshape(-1, 1)

days_in_future=20
future_forecast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates=future_forecast[:-20]

start='1/22/2020'
start_date=datetime.datetime.strptime(start, '%m/%d/%Y')
future_forecast_dates=[]
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed=train_test_split(days_since_1_22, world_cases, test_size=0.25, shuffle=False)

# Transformăm datele pentru regresia polinomială
poly=PolynomialFeatures(degree=3)
poly_X_train_confirmed=poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed=poly.fit_transform(X_test_confirmed)
poly_future_forecast=poly.fit_transform(future_forecast)

# Regresia polinomială
linear_model=LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred=linear_model.predict(poly_X_test_confirmed)
linear_pred=linear_model.predict(poly_future_forecast)

plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(['Date de testare', 'Predicții de regresie polinomială'])

adjusted_dates=adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, world_cases)
plt.title('Numărul de Cazuri Coronavirus Peste Timp', size=30)
plt.xlabel('Zile De La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, total_deaths)
plt.title('Numărul de Decese Coronavirus Peste Timp', size=30)
plt.xlabel('Zile De La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, total_recovered)
plt.title('Numărul de Recuperări Coronavirus Peste Timp', size=30)
plt.xlabel('Zile De La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, total_active)
plt.title('Numărul de Cazuri Active Coronavirus Peste Timp', size=30)
plt.xlabel('Zile De La 1/22/2020')
plt.ylabel('Numărul de Cazuri Active', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, world_daily_increase)
plt.title('Creșteri Mondiale Zilnice ale Cazurilor Confirmate', size=30)
plt.xlabel('Zile de La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, world_daily_death)
plt.title('Creșteri Mondiale Zilnice ale Deceselor Confirmate', size=30)
plt.xlabel('Zile De La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, world_daily_recovery)
plt.title('Creșteri Mondiale Zilnice ale Recuperărilor Confirmate', size=30)
plt.xlabel('Zile De La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

def plot_predictions(x, y, pred, algo_name, color):
    plt.figure(figsize=(16, 9))
    plt.plot(x, y)
    plt.plot(future_forecast, pred, linestyle='dashed', color=color)
    plt.title('Numărul de Cazuri Coronavirus Peste Timp', size=30)
    plt.xlabel('Zile De La 1/22/2020', size=30)
    plt.ylabel('Numărul de Cazuri', size=30)
    plt.legend(['Cazuri Confirmate', algo_name], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

plot_predictions(adjusted_dates, world_cases, linear_pred, 'Predicții de regresie polinomială', 'red')

# Predicții de viitor folosind regresia polinomială
linear_pred=linear_pred.reshape(1, -1)[0]
poly_df=pd.DataFrame({'Date': future_forecast_dates[-20:], 'Numărul prezis de Cazuri Mondiale Confirmate': np.round(linear_pred[-20:])})

# Predicții de viitor folosind SVM

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, total_deaths, color='r')
plt.plot(adjusted_dates, total_recovered, color='green')
plt.legend(['decese', 'recuperări'], loc='best', fontsize=20)
plt.title('Numărul de Cazuri Coronavirus', size=30)
plt.xlabel('Zile De La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(total_recovered, total_deaths)
plt.title('Numărul de Decese Coronavirus vs. Numărul de Recuperări Coronavirus', size=30)
plt.xlabel('Numărul de Recuperări Coronavirus', size=30)
plt.ylabel('Numărul de Decese Coronavirus', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

def country_plot(x, y1, y2, y3, y4, country):
    plt.figure(figsize=(16, 9))
    plt.plot(x, y1)
    plt.title('{} - Cazuri Confirmate'.format(country), size=30)
    plt.xlabel('Zile De La 1/22/2020', size=30)
    plt.ylabel('Numărul de Cazuri', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.bar(x, y2)
    plt.title('{} - Creșteri Zilnice ale Cazurilor Confirmate'.format(country), size=30)
    plt.xlabel('Zile De La 1/22/2020', size=30)
    plt.ylabel('Numărul de Cazuri', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.bar(x, y3)
    plt.title('{} - Creșteri Zilnice ale Deceselor'.format(country), size=30)
    plt.xlabel('Zile De La 1/22/2020', size=30)
    plt.ylabel('Numărul de Cazuri', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.bar(x, y4)
    plt.title('{} - Creșteri Zilnice ale Recuperărilor'.format(country), size=30)
    plt.xlabel('Zile De La 1/22/2020', size=30)
    plt.ylabel('Numărul de Cazuri', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

country_plot(adjusted_dates, china_cases, china_daily_increase, china_daily_death, china_daily_recovery, 'China')

country_plot(adjusted_dates, italy_cases, italy_daily_increase, italy_daily_death, italy_daily_recovery,  'Italia')

country_plot(adjusted_dates, india_cases, india_daily_increase, india_daily_death, india_daily_recovery, 'India')

country_plot(adjusted_dates, us_cases, us_daily_increase, us_daily_death, us_daily_recovery, 'Statele Unite')

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, china_cases)
plt.plot(adjusted_dates, italy_cases)
plt.plot(adjusted_dates, us_cases)
plt.plot(adjusted_dates, spain_cases)
plt.plot(adjusted_dates, france_cases)
plt.plot(adjusted_dates, germany_cases)
plt.plot(adjusted_dates, india_cases)
plt.title('Numărul de Cazuri Coronavirus', size=30)
plt.xlabel('Zile De La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.legend(['China', 'Italia', 'SUA', 'Spania', 'Franța', 'Germania', 'India'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, china_deaths)
plt.plot(adjusted_dates, italy_deaths)
plt.plot(adjusted_dates, us_deaths)
plt.plot(adjusted_dates, spain_deaths)
plt.plot(adjusted_dates, france_deaths)
plt.plot(adjusted_dates, germany_deaths)
plt.plot(adjusted_dates, india_deaths)
plt.title('Numărul de Decese Coronavirus', size=30)
plt.xlabel('Zile De La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.legend(['China', 'Italia', 'SUA', 'Spania', 'Franța', 'Germania', 'India'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, china_recoveries)
plt.plot(adjusted_dates, italy_recoveries)
plt.plot(adjusted_dates, us_recoveries)
plt.plot(adjusted_dates, spain_recoveries)
plt.plot(adjusted_dates, france_recoveries)
plt.plot(adjusted_dates, germany_recoveries)
plt.plot(adjusted_dates, india_recoveries)
plt.title('Numărul de Recuperări Coronavirus', size=30)
plt.xlabel('Zile De La 1/22/2020', size=30)
plt.ylabel('Numărul de Cazuri', size=30)
plt.legend(['China', 'Italia', 'SUA', 'Spania', 'Franța', 'Germania', 'India'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

