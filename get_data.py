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
        "cumPeopleVaccinatedSecondDoseByVaccinationDate":"cumPeopleVaccinatedSecondDoseByVaccinationDate"
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

    df = api.get_dataframe()
    df = df[['date', 'areaName', 'newPeopleVaccinatedFirstDoseByPublishDate', 
             'cumPeopleVaccinatedFirstDoseByPublishDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate',
            'newPeopleVaccinatedSecondDoseByPublishDate', 'cumPeopleVaccinatedSecondDoseByPublishDate', 
             'cumPeopleVaccinatedSecondDoseByVaccinationDate']]
    df.columns = ['date', 'area', 'daily_first_dose', 'cum_first_dose', 'cum_first_dose_weekly',
                 'daily_second_dose', 'cum_second_dose', 'cum_second_dose_weekly']
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")

    df = df.groupby('date').sum().reset_index()
    for row, col in df.iterrows():
        if col[2] == 0:
            df.loc[row, 'cum_first_dose'] = df.loc[row, 'cum_first_dose_weekly']
            df.loc[row, 'cum_second_dose'] = df.loc[row, 'cum_second_dose_weekly']

    df = df.append(pd.DataFrame([datetime.strptime("8 December, 2020", "%d %B, %Y")], 
                                columns=['date'])).fillna(0).sort_values('date')

    dates = pd.date_range(start=datetime.strptime("8 December, 2020", "%d %B, %Y"),
                        end=datetime.strptime("9 January, 2021", "%d %B, %Y"))
    dates = dates.to_frame()
    dates.columns = ['date']
    dates.reset_index(drop=True, inplace=True)
    df = dates.merge(df, on='date', how='outer').sort_values('date')

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
            if col[0] >= start_date+timedelta(weeks=12):
                if col[0] > end_date+timedelta(weeks=12):
                    df.loc[row, 'projection_second'] = df.loc[row-1, 'projection_second'] + \
                                                    (df.loc[row-84, 'projection_first'] - df.loc[row-85, 'projection_first'])
                    df.loc[row, 'projection_first'] = df.loc[row-1, 'projection_first'] + \
                                                    (df.loc[row-1, 'daily_rolling_average_total'] - \
                                                    (df.loc[row-84, 'projection_first'] - df.loc[row-85, 'projection_first']))
                else:
                    df.loc[row, 'projection_second'] = df.loc[row-1, 'projection_second'] + \
                                                    df.loc[row-84, 'daily_first_dose']
                    df.loc[row, 'projection_first'] = df.loc[row-1, 'projection_first'] + \
                                                    (df.loc[row-1, 'daily_rolling_average_total'] - \
                                                    df.loc[row-84, 'daily_first_dose'])
            else:
                df.loc[row, 'projection_first'] = df.loc[row-1, 'projection_first'] + \
                                                df.loc[row-1, 'daily_rolling_average_first']
                df.loc[row, 'projection_second'] = df.loc[row-1, 'projection_second'] + \
                                                    df.loc[row-1, 'daily_rolling_average_second']
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

    return df, last_update, rolling_avg, end_date


def make_cum_vaccine_plot(df, end_date):

    fig=px.line(df, x='date', y='cum_first_dose',
                labels={'date': 'Date (reported)', 'cum_first_dose': 'Cumulative Doses'})
    fig.update_traces(name='Total First', showlegend=True, line_color=colors['maincolor'], hovertemplate='%{y:,.0f}')

    sda = df.set_index('date').loc[end_date, 'daily_rolling_average_total']
    fig2 = px.line(df, x='date', y='projection_first')
    fig2.update_traces(name=f'Projection of first doses', 
                        showlegend=True, line_color=colors['maincolor'], 
                       hovertemplate='%{y:,.0f}', line=dict(dash='dash'))

    fig3=px.line(df, x='date', y='cum_second_dose',
                labels={'date': 'Date (reported)', 'cum_first_dose': 'Cumulative Second Doses'})
    fig3.update_traces(name='Total Second', showlegend=True, line_color=colors['monzo'], hovertemplate='%{y:,.0f}')

    fig4 = px.line(df, x='date', y='projection_second')
    fig4.update_traces(name=f'Projection of second doses', 
                        showlegend=True, line_color=colors['monzo'], 
                       hovertemplate='%{y:,.0f}', line=dict(dash='dash'))

    fig5=px.line(df, x='date', y='cum_immune',
                labels={'date': 'Date (reported)', 'cum_immune': 'Cumulative Immune'})
    fig5.update_traces(name='Total Immune', showlegend=True, line_color='purple', hovertemplate='%{y:,.0f}')

    fig6 = px.line(df, x='date', y='cum_immune_projected')
    fig6.update_traces(name=f'Projection of total immune', 
                        showlegend=True, line_color='purple', 
                       hovertemplate='%{y:,.0f}', line=dict(dash='dash'))

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
    fig=px.bar(df, x=x, y=y,labels={x: x_title, y: y_title})
    
    fig.update_traces(name=f'Daily Doses', marker_color=colors['maincolor'], hovertemplate='Daily: %{y:,.0f}', showlegend=True)

    fig2 = px.line(df, x=x, y='daily_rolling_average_total', line_shape='spline')
    fig2.update_traces(name=f'{rolling_avg} day rolling average', showlegend=True, line_color='#000000', hovertemplate='%{y:,.0f}')

    fig.add_trace(fig2.data[0])

    fig.update_layout(yaxis=dict(tickformat=',0.f', showgrid=False),xaxis=dict(showgrid=False),hovermode='x unified',showlegend= True,
                      font=dict(size=11),
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

    fig.update_layout(yaxis=dict(tickformat=',.0f'),xaxis=dict(showgrid=False),
                        font=dict(family="IBM Plex Sans", size=12, color="#000000"),hovermode="x unified")

    return fig


def make_daily_plot(df, x, y, x_title, y_title, log=False):
    fig = px.scatter(df, x=x, y=y, labels={x: x_title, y: y_title}, log_y=log)
    fig.update_traces(name=y_title, showlegend=True, marker_color=colors['maincolor'])

    fig2 = px.line(df, x=x, y='daily_rolling_average', line_shape='spline')
    fig2.update_traces(name='7 day rolling average', showlegend=True, line_color='#000000')
    
    
    mask = (df['date'] > datetime.strptime("7 January, 2021", "%d %B, %Y")) & (df['date'] <= datetime.strptime("31 January, 2021", "%d %B, %Y"))
    masked_df = df.loc[mask]

    x_vals = masked_df['date'].apply(lambda x: datetime.timestamp(x))
    y_vals = masked_df['daily_cases']
    
    popt, pcov = curve_fit(exponenial_func, x_vals, y_vals, p0=(-1, 0.01, 1))
    print(popt)
    
    to_plot = df.loc[df['date'] > datetime.strptime("7 January, 2021", "%d %B, %Y")]['date'].apply(lambda x: datetime.timestamp(x))
    x_to_plot = to_plot.apply(lambda x: datetime.fromtimestamp(x))
#     y_to_plot = [(slope*x)+intercept for x in to_plot]

    xx = x_to_plot
    yy = exponenial_func(to_plot, *popt)
    
    
    fig3 = px.line(x=xx, y=yy, labels={x: x_title, y: y_title})
    fig3.update_traces(name=y_title, showlegend=True, line_color=colors['monzo'])

    fig.add_trace(fig2.data[0])
    fig.add_trace(fig3.data[0])

    fig.update_layout(yaxis=dict(tickformat=',.0f'),font=dict(size=12, color="#000000"), showlegend=True,legend={"orientation": "h",
                            "xanchor": "center",
                            'y': 1.2,
                            'x': 0.5,
                            'font': dict(size=9)},hovermode="x unified")

    fig.update_traces(hovertemplate='%{y}')

    return fig

def exponenial_func(x, a, b, c):
    return a*np.exp(-b*x)+c