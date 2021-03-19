import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from datetime import datetime, timedelta
from uk_covid19 import Cov19API
import plotly.express as px
import plotly.io as pio
from pathlib import Path
import plotly.graph_objs as go
import plotly.graph_objects as go


# Plotly settings
pio.templates.default = 'plotly_white'
pd.set_option('display.max_columns', None)
colors = {'maincolor': '#3269a8', 'monzo': '#f88379', 'lloyds': '#024731'}


def get_vaccinations(df):
    
    df = df[['date', 'areaName', 'newPeopleVaccinatedFirstDoseByPublishDate', 
             'cumPeopleVaccinatedFirstDoseByPublishDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate',
            'newPeopleVaccinatedSecondDoseByPublishDate', 'cumPeopleVaccinatedSecondDoseByPublishDate', 
             'cumPeopleVaccinatedSecondDoseByVaccinationDate']]
    df.columns = ['date', 'area', 'daily_first_dose', 'cum_first_dose', 'cum_first_dose_weekly',
                 'daily_second_dose', 'cum_second_dose', 'cum_second_dose_weekly']
    df.loc[:, 'date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df[df['date']>=datetime.strptime("8 December, 2020", "%d %B, %Y")]

    df = df.groupby('date').sum().reset_index()
    df = df[:-1]
    for row, col in df.iterrows():
        if row != 0:
            if col[2] == 0:
                if col[3] != 0:
                    df.loc[row, 'cum_first_dose'] = df.loc[row, 'cum_first_dose_weekly']
                    df.loc[row, 'cum_second_dose'] = df.loc[row, 'cum_second_dose_weekly']
                else:
                    df.loc[row, 'cum_first_dose'] = np.nan
                    df.loc[row, 'cum_second_dose'] = np.nan
                
    df['cum_first_dose'] = df['cum_first_dose'].interpolate(method='linear')
    df['cum_second_dose'] = df['cum_second_dose'].interpolate(method='linear')
    
    for row, col in df.iterrows():
        if row != 0:
            if col[1] == 0 or np.isnan(col[1]):
                df.loc[row, 'daily_first_dose'] = df.loc[row, 'cum_first_dose']/row
                df.loc[row, 'daily_second_dose'] = df.loc[row, 'cum_second_dose']/row

    df['daily_total_doses'] = df['daily_first_dose'] + df['daily_second_dose']
    df['cum_total_doses'] = df['cum_first_dose'] + df['cum_second_dose']
    
    rolling_avg=7
    df['daily_rolling_average_first'] = df['daily_first_dose'].rolling(window=rolling_avg, 
                                                                 min_periods=rolling_avg).mean()
    df['daily_rolling_average_second'] = df['daily_second_dose'].rolling(window=rolling_avg, 
                                                                 min_periods=rolling_avg).mean()
    df['daily_rolling_average_total'] = df['daily_total_doses'].rolling(window=rolling_avg, 
                                                                 min_periods=rolling_avg).mean()
    start_date = df['date'].min() + +timedelta(days=1)
    end_date = df['date'].max()
    dates = pd.date_range(start=end_date+timedelta(days=1),
                        end=end_date+timedelta(days=150))
    dates = dates.to_frame()
    dates.columns = ['date']
    dates['projection_first'] = 0
    dates['projection_second'] = 0
    dates['cum_immune'] = 0
    dates['cum_immune_projected'] = 0
    dates.reset_index(drop=True, inplace=True)
    df = dates.merge(df, on='date', how='outer').sort_values('date')
    df = df.reset_index(drop='index')

    for row, col in df.iterrows():
        if col[0] == end_date:
            df.loc[row, 'projection_first'] = df.loc[row, 'cum_first_dose']
            df.loc[row, 'projection_second'] = df.loc[row, 'cum_second_dose']
        if col[0] > end_date:
            df.loc[row, 'daily_rolling_average_first'] = df.loc[row-1, 'daily_rolling_average_first']
            df.loc[row, 'daily_rolling_average_second'] = df.loc[row-1, 'daily_rolling_average_second']
            df.loc[row, 'daily_rolling_average_total'] = df.loc[row-1, 'daily_rolling_average_total']
            if col[0] >= start_date+timedelta(weeks=11):
                if col[0] > end_date+timedelta(weeks=11):
                    df.loc[row, 'projection_second'] = df.loc[row-1, 'projection_second'] + \
                                                    max((df.loc[row-77, 'projection_first'] - df.loc[row-78, 'projection_first']), 0)
                    df.loc[row, 'projection_first'] = df.loc[row-1, 'projection_first'] + \
                                                    max((df.loc[row, 'daily_rolling_average_total'] - \
                                                    (df.loc[row-77, 'projection_first'] - df.loc[row-78, 'projection_first'])), 0)
                else:
                    df.loc[row, 'projection_second'] = df.loc[row-1, 'projection_second'] + \
                                                    max(df.loc[row-77, 'daily_first_dose'], 0)
                    df.loc[row, 'projection_first'] = df.loc[row-1, 'projection_first'] + \
                                                    max((df.loc[row, 'daily_rolling_average_total'] - \
                                                    df.loc[row-77, 'daily_first_dose']), 0)
            else:
                df.loc[row, 'projection_first'] = df.loc[row-1, 'projection_first'] + \
                                                df.loc[row, 'daily_rolling_average_first']
                df.loc[row, 'projection_second'] = df.loc[row-1, 'projection_second'] + \
                                                    df.loc[row, 'daily_rolling_average_second']
    for row, col in df.iterrows():
        if col[0] < (end_date + timedelta(days=-21)):
            df.loc[row+21, 'cum_immune'] = df.loc[row, 'cum_second_dose']
        elif col[0] == (end_date + timedelta(days=-21)):
            df.loc[row+21, 'cum_immune'] = df.loc[row, 'cum_second_dose']
            df.loc[row+21, 'cum_immune_projected'] = df.loc[row, 'cum_second_dose']
        elif col[0] > (end_date + timedelta(days=-21)) and col[0] < end_date:
            df.loc[row+21, 'cum_immune'] = np.nan
            df.loc[row+21, 'cum_immune_projected'] = df.loc[row, 'cum_second_dose']
        else:
            df.loc[row+21, 'cum_immune'] = np.nan
            df.loc[row+21, 'cum_immune_projected'] = df.loc[row, 'projection_second']
            
    return df, rolling_avg, end_date


def get_cases(df):
    
    df = df[['date', 'areaName', 'newCasesByPublishDate']]
    df.columns = ['date', 'area', 'daily_cases']
    df.loc[:, 'date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.groupby('date').sum().reset_index()

    df['daily_rolling_average'] = df['daily_cases'].rolling(window=7, min_periods=7, center=True).mean()
    df = df.dropna()
    df['days_since_start'] = df['date'].apply(lambda x: (x - datetime(2021,1,9)).days)

    recent_df = df.copy()
    recent_df = recent_df[recent_df['date']>datetime.strptime("9 January, 2021", "%d %B, %Y")]

    fit_end = datetime.strptime("17 February, 2021", "%d %B, %Y")
    pars, cov = curve_fit(f=exponential, 
                          xdata=recent_df[recent_df['date']<fit_end]['days_since_start'], 
                          ydata=recent_df[recent_df['date']<fit_end]['daily_rolling_average'], maxfev=1000)
    recent_df.loc[:, 'fit'] = recent_df['days_since_start'].apply(lambda x: exponential(x, *pars))
    return recent_df, fit_end


def get_deaths(df):
    
    df = df[['date', 'areaName', 'newDeaths28DaysByPublishDate']]
    df.columns = ['date', 'area', 'daily_deaths']
    df.loc[:, 'date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.groupby('date').sum().reset_index()

    df['daily_rolling_average'] = df['daily_deaths'].rolling(window=7, min_periods=7, center=True).mean()
    df = df.dropna()
    df['days_since_start'] = df['date'].apply(lambda x: (x - datetime(2021,1,30)).days)

    recent_df = df.copy()
    recent_df = recent_df[recent_df['date']>datetime.strptime("30 January, 2021", "%d %B, %Y")]

    fit_end = datetime.strptime("22 February, 2021", "%d %B, %Y")
    pars, cov = curve_fit(f=exponential, xdata=recent_df[recent_df['date']<fit_end]['days_since_start'], 
                              ydata=recent_df[recent_df['date']<fit_end]['daily_rolling_average'], maxfev=1000)
    recent_df.loc[:, 'fit'] = recent_df['days_since_start'].apply(lambda x: exponential(x, *pars))
    return recent_df, fit_end


def get_admissions(df):
    
    df = df[['date', 'areaName', 'newAdmissions']]
    df.columns = ['date', 'area', 'daily_admissions']
    df.loc[:, 'date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df = df.groupby('date').sum().reset_index()

    df['daily_rolling_average'] = df['daily_admissions'].rolling(window=7, min_periods=7, center=True).mean()
    df = df.dropna()
    df['days_since_start'] = df['date'].apply(lambda x: (x - datetime(2021,1,22)).days)

    recent_df = df.copy()
    recent_df = recent_df[recent_df['date']>datetime.strptime("22 January, 2021", "%d %B, %Y")]
    recent_df = recent_df.iloc[:-7]

    fit_end = datetime.strptime("22 February, 2021", "%d %B, %Y")
    pars, cov = curve_fit(f=exponential, xdata=recent_df[recent_df['date']<fit_end]['days_since_start'], 
                              ydata=recent_df[recent_df['date']<fit_end]['daily_rolling_average'], maxfev=1000)
    recent_df.loc[:, 'fit'] = recent_df['days_since_start'].apply(lambda x: exponential(x, *pars))
    return recent_df, fit_end


def get_admissions_by_age(df):
    
    age_groups = {
        'under_65s': ['0_to_5', '6_to_17', '18_to_64'],
        'over_65s': ['65_to_84', '85+']
    }
    
    df = df[['date', 'areaName', 'cumAdmissionsByAge']]
    admissions_by_age = {}
    for row, col in df.iterrows():
        if col[2]:
            admissions_by_age[col[0]] = {}
            for i in col[2]:
                admissions_by_age[col[0]][i['age']] = i['rate']

    admissions_by_age_df = pd.DataFrame(admissions_by_age)
    admissions_by_age_df = admissions_by_age_df.T.reset_index().rename(columns={'index':'date'}).sort_values('date').reset_index(drop=True)
    admissions_by_age_df.loc[:, 'date'] = pd.to_datetime(admissions_by_age_df['date'], format="%Y-%m-%d")

    for group in age_groups.keys():
        admissions_by_age_df[group] = admissions_by_age_df[age_groups[group]].sum(axis=1)
    cols = ['date']
    cols.extend(list(age_groups.keys()))
    admissions_by_age_df = admissions_by_age_df[cols]

    for group in age_groups.keys():
        admissions_by_age_df[f'{group}_daily'] = 0

    for row, col in admissions_by_age_df.iterrows():
        for i in range(1, len(age_groups.keys())+1):
            if row == 0:
                admissions_by_age_df.iloc[row, i+len(age_groups.keys())] = admissions_by_age_df.iloc[row, i]
            else:
                admissions_by_age_df.iloc[row, i+len(age_groups.keys())] = admissions_by_age_df.iloc[row, i] - admissions_by_age_df.iloc[row-1, i]

    for group in age_groups.keys():
        admissions_by_age_df[f'{group}_daily_rolling_average'] = admissions_by_age_df[f'{group}_daily'].rolling(window=7, 
                                                                         min_periods=7, center=True).mean()
        max_val = admissions_by_age_df[f'{group}_daily_rolling_average'].max()
        admissions_by_age_df[f'{group}_indexed'] = admissions_by_age_df[f'{group}_daily_rolling_average'].apply(lambda x: 100*x/max_val)

    admissions_by_age_df = admissions_by_age_df[admissions_by_age_df['date']>datetime(2020,12,1)].reset_index(drop=True)
    cols_to_keep = [f'{group}_indexed' for group in age_groups.keys()]
    cols_to_keep.append('date')
    admissions_by_age_df = admissions_by_age_df[cols_to_keep]
    return admissions_by_age_df, age_groups


def get_cases_by_age(df):
    
    age_groups = {
        'under_65s': ['0_to_4', '5_to_9', '10_to_14', '15_to_19', '20_to_24',
                       '25_to_29', '30_to_34', '35_to_39', '40_to_44', '45_to_49', '50_to_54',
                       '55_to_59', '60_to_64'],
        'over_65s': ['65_to_69', '70_to_74', '75_to_79', '80_to_84', '85_to_89', '90+']
    }
    
    df = df[['date', 'areaName', 'maleCases', 'femaleCases']]
    cases_by_age_male = {}
    cases_by_age_female = {}
    for row, col in df.iterrows():
        if col[2]:
            cases_by_age_male[col[0]] = {}
            for i in col[2]:
                cases_by_age_male[col[0]][i['age']] = i['value']
        if col[3]:
            cases_by_age_female[col[0]] = {}
            for i in col[3]:
                cases_by_age_female[col[0]][i['age']] = i['value']
        

    cases_by_age_male_df = pd.DataFrame(cases_by_age_male)
    cases_by_age_male_df = cases_by_age_male_df.T.reset_index().rename(columns={'index':'date'}).sort_values('date').reset_index(drop=True)
    cases_by_age_male_df = cases_by_age_male_df.groupby('date').sum().reset_index()
    cases_by_age_female_df = pd.DataFrame(cases_by_age_female)
    cases_by_age_female_df = cases_by_age_female_df.T.reset_index().rename(columns={'index':'date'}).sort_values('date').reset_index(drop=True)
    cases_by_age_female_df = cases_by_age_female_df.groupby('date').sum().reset_index()
    
    cases_by_age_df = cases_by_age_male_df.set_index('date') + cases_by_age_male_df.set_index('date')
    cases_by_age_df = cases_by_age_df.reset_index()
    cases_by_age_df.loc[:, 'date'] = pd.to_datetime(cases_by_age_df['date'], format="%Y-%m-%d")
    cases_by_age_df = cases_by_age_df.iloc[:-4]

    for group in age_groups.keys():
        cases_by_age_df[group] = cases_by_age_df[age_groups[group]].sum(axis=1)
    cols = ['date']
    cols.extend(list(age_groups.keys()))
    cases_by_age_df = cases_by_age_df[cols]

    for group in age_groups.keys():
        cases_by_age_df[f'{group}_daily'] = 0

    for row, col in cases_by_age_df.iterrows():
        for i in range(1, len(age_groups.keys())+1):
            if row == 0:
                cases_by_age_df.iloc[row, i+len(age_groups.keys())] = cases_by_age_df.iloc[row, i]
            else:
                cases_by_age_df.iloc[row, i+len(age_groups.keys())] = cases_by_age_df.iloc[row, i] - cases_by_age_df.iloc[row-1, i]

    for group in age_groups.keys():
        cases_by_age_df[f'{group}_daily_rolling_average'] = cases_by_age_df[f'{group}_daily'].rolling(window=7, 
                                                                         min_periods=7, center=True).mean()
        max_val = cases_by_age_df[f'{group}_daily_rolling_average'].max()
        cases_by_age_df[f'{group}_indexed'] = cases_by_age_df[f'{group}_daily_rolling_average'].apply(lambda x: 100*x/max_val)

    cases_by_age_df = cases_by_age_df[cases_by_age_df['date']>datetime(2020,12,1)].reset_index(drop=True)
    cols_to_keep = [f'{group}_indexed' for group in age_groups.keys()]
    cols_to_keep.append('date')
    cases_by_age_df = cases_by_age_df[cols_to_keep]
    return cases_by_age_df, age_groups


def get_deaths_by_age(df):
    
    age_groups = {
        'under_65s': ['00_04', '10_14', '15_19', '20_24', '25_29', '30_34', '35_39',
                       '40_44', '45_49', '50_54', '55_59', '05_09', '60_64'],
        'over_65s': ['65_69', '70_74', '75_79', '80_84', '85_89', '90+']
    }
    
    df = df[['date', 'areaName', 'newDeaths28DaysByDeathDateAgeDemographics']]
    df = df[df['areaName'] == 'England']
    deaths_by_age = {}
    for row, col in df.iterrows():
        if col[2]:
            deaths_by_age[col[0]] = {}
            for i in col[2]:
                deaths_by_age[col[0]][i['age']] = i['deaths']
        
    deaths_by_age_df = pd.DataFrame(deaths_by_age)
    deaths_by_age_df = deaths_by_age_df.T.reset_index().rename(columns={'index':'date'}).sort_values('date').reset_index(drop=True)
    deaths_by_age_df = deaths_by_age_df.groupby('date').sum().reset_index()

    deaths_by_age_df.loc[:, 'date'] = pd.to_datetime(deaths_by_age_df['date'], format="%Y-%m-%d")

    for group in age_groups.keys():
        deaths_by_age_df[group] = deaths_by_age_df[age_groups[group]].sum(axis=1)
    cols = ['date']
    cols.extend(list(age_groups.keys()))
    deaths_by_age_df = deaths_by_age_df[cols]

    for group in age_groups.keys():
        deaths_by_age_df[f'{group}_daily_rolling_average'] = deaths_by_age_df[f'{group}'].rolling(window=7, 
                                                                         min_periods=7, center=True).mean()
        max_val = deaths_by_age_df[f'{group}_daily_rolling_average'].max()
        deaths_by_age_df[f'{group}_indexed'] = deaths_by_age_df[f'{group}_daily_rolling_average'].apply(lambda x: 100*x/max_val)

    deaths_by_age_df = deaths_by_age_df[deaths_by_age_df['date']>datetime(2020,12,1)].reset_index(drop=True)
    cols_to_keep = [f'{group}_indexed' for group in age_groups.keys()]
    cols_to_keep.append('date')
    deaths_by_age_df = deaths_by_age_df[cols_to_keep]
    return deaths_by_age_df, age_groups


def get_covid_data():
    all_nations = [
    "areaType=nation"
    ]

    get_data = {
        "date": "date",
        "areaName": "areaName",
        "newPeopleVaccinatedFirstDoseByPublishDate":"newPeopleVaccinatedFirstDoseByPublishDate",
        "newPeopleVaccinatedSecondDoseByPublishDate":"newPeopleVaccinatedSecondDoseByPublishDate",
        "cumPeopleVaccinatedFirstDoseByPublishDate":"cumPeopleVaccinatedFirstDoseByPublishDate",
        "cumPeopleVaccinatedSecondDoseByPublishDate":"cumPeopleVaccinatedSecondDoseByPublishDate",
        "weeklyPeopleVaccinatedFirstDoseByVaccinationDate":"weeklyPeopleVaccinatedFirstDoseByVaccinationDate",
        "cumPeopleVaccinatedFirstDoseByVaccinationDate":"cumPeopleVaccinatedFirstDoseByVaccinationDate",
        "weeklyPeopleVaccinatedSecondDoseByVaccinationDate":"weeklyPeopleVaccinatedSecondDoseByVaccinationDate",
        "cumPeopleVaccinatedSecondDoseByVaccinationDate":"cumPeopleVaccinatedSecondDoseByVaccinationDate",
        "newCasesByPublishDate": "newCasesByPublishDate",
        "newDeaths28DaysByPublishDate": "newDeaths28DaysByPublishDate",
        "newAdmissions": "newAdmissions",
        "cumAdmissionsByAge": "cumAdmissionsByAge",
        "maleCases": "maleCases",
        "femaleCases": "femaleCases",
        "newDeaths28DaysByDeathDateAgeDemographics": "newDeaths28DaysByDeathDateAgeDemographics"
    }

    api = Cov19API(
        filters=all_nations,
        structure=get_data,
        latest_by="newPeopleVaccinatedFirstDoseByPublishDate"
    )

    last_update = api.get_json()['lastUpdate']
    last_update = datetime.fromisoformat(last_update[:-1])
    last_update = last_update.strftime("%a %d %b %H:%M")

    api = Cov19API(
        filters=all_nations,
        structure=get_data
    )

    full_df = api.get_dataframe()
    
    recent_cases_df, recent_cases_fit_end = get_cases(full_df)
    cases_by_age_df, cases_age_groups = get_cases_by_age(full_df)
    recent_deaths_df, recent_deaths_fit_end = get_deaths(full_df)
    deaths_by_age_df, deaths_age_groups = get_deaths_by_age(full_df)
    recent_admissions_df, recent_admissions_fit_end = get_admissions(full_df)
    admissions_by_age_df, ad_age_groups  = get_admissions_by_age(full_df)
    vacc_df, rolling_avg, end_date = get_vaccinations(full_df)
            
    return vacc_df, last_update, rolling_avg, end_date, recent_cases_df, recent_cases_fit_end, \
            recent_deaths_df, recent_deaths_fit_end, recent_admissions_df, recent_admissions_fit_end, \
            admissions_by_age_df, ad_age_groups, cases_by_age_df, cases_age_groups, \
            deaths_by_age_df, deaths_age_groups


def make_cum_vaccine_plot(df, end_date):

    fig=px.line(df, x='date', y='cum_first_dose',
                labels={'date': 'Date (reported)', 'cum_first_dose': 'Cumulative Doses'})
    fig.update_traces(name='Total First Doses', showlegend=True, line_color=colors['maincolor'], 
                      hovertemplate='%{y:,.0f}', line=dict(width=3))

    sda = df.set_index('date').loc[end_date, 'daily_rolling_average_total']
    fig2 = px.line(df, x='date', y='projection_first')
    fig2.update_traces(name=f'Projection of first doses', 
                        showlegend=True, line_color=colors['maincolor'], 
                       hovertemplate='%{y:,.0f}', line=dict(dash='dash', width=3))

    fig3=px.line(df, x='date', y='cum_second_dose',
                labels={'date': 'Date (reported)', 'cum_first_dose': 'Cumulative Second Doses'})
    fig3.update_traces(name='Total Second Doses', showlegend=True, line_color=colors['monzo'], 
                       hovertemplate='%{y:,.0f}', line=dict(width=3))

    fig4 = px.line(df, x='date', y='projection_second')
    fig4.update_traces(name=f'Projection of second doses', 
                        showlegend=True, line_color=colors['monzo'], 
                       hovertemplate='%{y:,.0f}', line=dict(dash='dash', width=3))

    fig5=px.line(df, x='date', y='cum_immune',
                labels={'date': 'Date (reported)', 'cum_immune': 'Cumulative Immune'})
    fig5.update_traces(name='Total Immune', showlegend=True, line_color='purple', 
                       hovertemplate='%{y:,.0f}', line=dict(width=3), visible = "legendonly")

    fig6 = px.line(df, x='date', y='cum_immune_projected')
    fig6.update_traces(name=f'Projection of total immune', 
                        showlegend=True, line_color='purple', 
                       hovertemplate='%{y:,.0f}', line=dict(dash='dash', width=3), visible = "legendonly")

    fig.add_trace(fig3.data[0])
    fig.add_trace(fig5.data[0])
    fig.add_trace(fig2.data[0])
    fig.add_trace(fig4.data[0])
    fig.add_trace(fig6.data[0])

    fig.update_layout(yaxis=dict(tickformat=',0.f', showgrid=False),
                      xaxis=dict(showgrid=False),
                      hovermode='x unified',
                      showlegend= True,
                      font=dict(size=11),
                      legend={"orientation": "h",
                              "xanchor": "center",
                              'x': 0.5,
                              'y': -0.2,
                              'font': dict(size=9)
                             },
                        shapes=[
                            dict(type= 'line',  layer='below',
                                yref='y',
                                y0=13.4e6, y1=13.4e6,
                                xref='x', 
                                x0=datetime.strptime("8 December, 2020", "%d %B, %Y"),
                                x1=df['date'].max() + timedelta(days=20),
                                line=dict(dash="dot", color=colors['lloyds'])),
                            dict(type= 'line', layer='below',
                                yref='y',
                                y0=29.6e6, y1=29.6e6,
                                xref='x', 
                                x0=datetime.strptime("8 December, 2020", "%d %B, %Y"),
                                x1=df['date'].max() + timedelta(days=20),
                                line=dict(dash="dot", color=colors['lloyds']))
                            ])

    fig.add_annotation(x=0, xref='paper', yref='y', y=13.4e6, ax=25, ay=25, align="left",
            text="Top 4 Priority Groups", xanchor="left", yanchor='bottom', xshift=5, showarrow=False, arrowhead=1, arrowsize = 2)
    fig.add_annotation(x=0, xref='paper', yref='y', y=29.6e6, ax=40, ay=0, yanchor='bottom', align="left",
            text="Priority Groups<br>1-9 (all over 50s)", xanchor="left", xshift=5, showarrow=False, arrowhead=1, arrowsize = 2)

    fig.update_xaxes(range=[datetime.strptime("8 December, 2020", "%d %B, %Y"), 
                            df['date'].max() + timedelta(days=20)], 
                            scaleratio = 0.1)
    return fig


def make_bar(df, x_title, y_title, x, y, rolling_avg):
    fig=px.bar(df, x=x, y=y,labels={x: x_title, 'value': y_title},
               color_discrete_sequence=[colors['maincolor'], colors['monzo']])
        
    fig.update_traces(hovertemplate='%{y:,.0f}', showlegend=True)

    fig2 = px.line(df, x=x, y='daily_rolling_average_total', line_shape='spline',
                        custom_data=["daily_total_doses"],)
    fig2.update_traces(name=f'{rolling_avg} day rolling average (trailing)', showlegend=True, line_color='#000000',
                        hovertemplate='%{y:,.0f}<br>Total Daily Doses: %{customdata[0]:,.0f}',
                        line=dict(width=3))

    fig.add_trace(fig2.data[0])

    fig.update_layout(yaxis=dict(tickformat=',0.f', showgrid=False),xaxis=dict(showgrid=False),
                      hovermode='x unified',showlegend= True,
                      font=dict(size=11),legend_title_text=None,
                            legend={"orientation": "h",
                              "xanchor": "center",
                              'x': 0.5,
                              'y': -0.2,
                              'font': dict(size=11)
                             })

    fig.update_xaxes(range=[datetime.strptime("11 January, 2021", "%d %B, %Y") + timedelta(hours=-12), 
                            df[x].max() + timedelta(days=-150, hours=12)], 
                            scaleratio = 0.1)
    return fig


def make_gauge(num_vax, previous_vax):
    
    pop = 66.65e6
    group_pops = [0, 1, 3, 2.5, 2.2, 3.3, 1.4, 3.3, 3.8, 4.4, 4.7, 37.05]
    group_pops = [x*1e6 for x in group_pops]
    group_prop = np.cumsum(group_pops)
    groups = ["Care Homes", "80+", "Frontline Health Workers", "75+", "70+",
                "Clinically Vulnerable", "65+","60+", "55+", "50+", "Everyone Else", "Entire Population"]
    colours = ["012a4a","013a63","01497c","014f86","2a6f97","2c7da0","468faf","61a5c2","89c2d9","a9d6e5","c7e2eb"]
    colours = ['#'+i for i in colours]
    
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = num_vax,
        number = {'valueformat':',.0f'},
        mode = "gauge+number+delta",
        title = {'text': "UK Vaccine Rollout"},
        delta = {'reference': previous_vax, 'valueformat':',.0f'},
        gauge = {'axis': {'range': [None, pop], 
                          'tickvals':group_prop,
                         'ticktext':groups},
                 'steps': 
                 [{'range': [group_prop[i], group_prop[i+1]], 
                 'color': colours[i]} for i in range((len(group_pops)-1))]},
        name='Cumulative Number of First Doses'))

    return fig


def make_cumulative_plot(df, x, y, x_title, y_title):
    fig = px.line(df, x=x, y=y, line_shape='spline',labels={x: x_title,y: y_title})

    fig.update_traces(line_color=colors['maincolor'], hovertemplate='%{y}')

    fig.update_layout(yaxis=dict(tickformat=',.0f', showgrid=False), xaxis=dict(showgrid=False),
                        font=dict(family="IBM Plex Sans", size=12, color="#000000"),hovermode="x unified")

    return fig


def make_7da_plot(df, log=False, metric='', fit_end=datetime.now()+timedelta(days=-1)):
    fig = px.scatter(df, x='date', y='daily_rolling_average', 
                     labels={'date': 'Date (reported)', 
                             'daily_rolling_average': f'Daily {metric}'}, 
                     log_y=log)
    fig.update_traces(name=f'{metric} (rolling weekly average)', 
                      showlegend=True, marker_color=colors['maincolor'])
    
    fig2 = px.line(df, x='date', y='fit')
    fig2.update_traces(name=f'Fit (on data up to {fit_end.strftime("%d %B")})', showlegend=True, line_color='#000000', line=dict(width=3))
    
    fig.add_trace(fig2.data[0])

    fig.update_layout(yaxis=dict(tickformat=',.0f', showgrid=False),
                      xaxis=dict(showgrid=False),
                      font=dict(size=12, color="#000000"), 
                      showlegend=True,
                      legend={"orientation": "h",
                              "xanchor": "center",
                              'x': 0.5,
                              'y': -0.2,
                              'font': dict(size=11)
                             },
                      hovermode="x unified")

    fig.update_traces(hovertemplate='%{y}')

    return fig


def make_indexed_plot(df, groups, log=True, metric=''):
    groups.append('date')
    df.columns = groups
    fig = px.line(df, x='date', y=groups, log_y=log, 
                  color_discrete_sequence=[colors['maincolor'], '#fe7f9c', '#0b6623'],
                  labels={'date': 'Date','value': f'Daily {metric} - percentage of winter peak'})
    fig.update_traces(line=dict(width=3))

    fig.update_layout(yaxis=dict(ticksuffix='%', tickformat=',.0f', tickmode = 'array',
                    tickvals = [1, 10, 20, 50, 70, 100], showgrid=False),
                      xaxis=dict(showgrid=False),
                      font=dict(size=12, color="#000000"), 
                      showlegend=True,
                      legend={"orientation": "h",
                              "xanchor": "center",
                              'x': 0.5,
                              'y': -0.2,
                              'font': dict(size=11)
                             },
                      legend_title_text=None,
                      hovermode="x unified")

    fig.update_traces(hovertemplate='%{y}')

    return fig


def exponential(x, a, b):
    return a*np.exp(b*x)