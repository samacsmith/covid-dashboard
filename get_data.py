import pandas as pd
import numpy as np
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
    df = df[['date', 'areaName', 'newPeopleVaccinatedFirstDoseByPublishDate', 'cumPeopleVaccinatedFirstDoseByPublishDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate']]
    df.columns = ['date', 'area', 'daily_first_dose', 'cum_first_dose', 'cum_first_dose_weekly']
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    
    df = df.groupby('date').sum().reset_index()
    for row, col in df.iterrows():
        if col[2] == 0:
            df.loc[row, 'cum_first_dose'] = df.loc[row, 'cum_first_dose_weekly']

    df = df.append(pd.DataFrame([datetime.strptime("8 December, 2020", "%d %B, %Y")], columns=['date'])).fillna(0).sort_values('date')

    dates = pd.date_range(start=datetime.strptime("8 December, 2020", "%d %B, %Y"),
                        end=datetime.strptime("9 January, 2021", "%d %B, %Y"))
    dates = dates.to_frame()
    dates.columns = ['date']
    dates.reset_index(drop=True, inplace=True)
    df = dates.merge(df, on='date', how='outer').sort_values('date')

    df['cum_first_dose'] = df['cum_first_dose'].interpolate(method='linear')
    
    for row, col in df.iterrows():
        if row != 0:
            if col[1] == 0 or np.isnan(col[1]):
                df.loc[row, 'daily_first_dose'] = df.loc[row, 'cum_first_dose']/row
    
    rolling_avg=7
    df['daily_rolling_average'] = df['daily_first_dose'].rolling(window=rolling_avg, min_periods=rolling_avg).mean()
    
    end_date = df['date'].max()
    dates = pd.date_range(start=end_date+timedelta(days=1),
                        end=end_date+timedelta(days=90))
    dates = dates.to_frame()
    dates.columns = ['date']
    dates['projection'] = 0
    dates['projection_4m'] = 0
    dates['projection_3m'] = 0
    dates['cum_immune'] = 0
    dates['cum_immune_projected'] = 0
    dates['cum_21_days'] = 0
    dates['cum_21_days_projected'] = 0
    dates.reset_index(drop=True, inplace=True)
    df = dates.merge(df, on='date', how='outer').sort_values('date')
    df = df.reset_index(drop='index')
    df['daily_rolling_average_projection'] = df['daily_rolling_average']
    
    for row, col in df.iterrows():
        if col[0] == end_date:
            df.loc[row, 'projection'] = df.loc[row, 'cum_first_dose']
            df.loc[row, 'projection_4m'] = df.loc[row, 'cum_first_dose']
            df.loc[row, 'projection_3m'] = df.loc[row, 'cum_first_dose']
        if col[0] > end_date:
            df.loc[row, 'projection'] = df.loc[row-1, 'projection'] + df.loc[row-1, 'daily_rolling_average_projection']
            df.loc[row, 'daily_rolling_average_projection'] = df.loc[row-1, 'daily_rolling_average_projection']
            df.loc[row, 'projection_4m'] = df.loc[row-1, 'projection_4m'] + (4e6/7)
            df.loc[row, 'projection_3m'] = df.loc[row-1, 'projection_3m'] + (3e6/7)
    
    for row, col in df.iterrows():
        if col[0] < (end_date + timedelta(days=-21)):
            df.loc[row+21, 'cum_immune'] = 0.7 * df.loc[row, 'cum_first_dose']
            df.loc[row+21, 'cum_21_days'] = df.loc[row, 'cum_first_dose']
        elif col[0] == (end_date + timedelta(days=-21)):
            df.loc[row+21, 'cum_immune'] = 0.7 * df.loc[row, 'cum_first_dose']
            df.loc[row+21, 'cum_immune_projected'] = 0.7 * df.loc[row, 'cum_first_dose']
            df.loc[row+21, 'cum_21_days'] = df.loc[row, 'cum_first_dose']
            df.loc[row+21, 'cum_21_days_projected'] = df.loc[row, 'cum_first_dose']
        elif col[0] > (end_date + timedelta(days=-21)) and col[0] < end_date:
            df.loc[row+21, 'cum_immune'] = np.nan
            df.loc[row+21, 'cum_immune_projected'] = 0.7 * df.loc[row, 'cum_first_dose']
            df.loc[row+21, 'cum_21_days'] = np.nan
            df.loc[row+21, 'cum_21_days_projected'] = df.loc[row, 'cum_first_dose']
        else:
            df.loc[row+21, 'cum_immune'] = np.nan
            df.loc[row+21, 'cum_immune_projected'] = 0.7 * df.loc[row, 'projection']
            df.loc[row+21, 'cum_21_days'] = np.nan
            df.loc[row+21, 'cum_21_days_projected'] = df.loc[row, 'projection']
        
    return df, last_update, rolling_avg, end_date



def make_plot(df, x_title, y_title, x, y, end_date):
    fig=px.line(df, x=x, y=y,
            labels={x: x_title, y: y_title})
    fig.update_traces(name='Total', showlegend=True, line_color=colors['maincolor'], hovertemplate='%{y:,.0f}')

    sda = df.set_index('date').loc[end_date, 'daily_rolling_average']
    fig2 = px.line(df, x=x, y='projection')
    fig2.update_traces(name=f'Projection (7 day average - {sda:,.0f})', 
                        showlegend=True, line_color=colors['monzo'], hovertemplate='%{y:,.0f}', visible="legendonly")
    fig3 = px.line(df, x=x, y='projection_4m')
    fig3.update_traces(name='Projection (4m a week)', showlegend=True, line_color=colors['lloyds'], hovertemplate='%{y:,.0f}', visible="legendonly")
    fig4 = px.line(df, x=x, y='projection_3m')
    fig4.update_traces(name='Projection (3m a week)', showlegend=True, line_color=colors['lloyds'], hovertemplate='%{y:,.0f}', visible="legendonly")
    
    fig5=px.line(df, x=x, y='cum_immune')
    fig5.update_traces(name='Total Immune', showlegend=True, line_color='red', hovertemplate='%{y:,.0f}')
    fig6=px.line(df, x=x, y='cum_immune_projected')
    fig6.update_traces(name='Projected Total Immune', showlegend=True, line_color='orange', hovertemplate='%{y:,.0f}', visible="legendonly")
    fig7=px.line(df, x=x, y='cum_21_days')
    fig7.update_traces(name='Total 21 days after Vaccination', showlegend=True, line_color='purple', hovertemplate='%{y:,.0f}')
    fig8=px.line(df, x=x, y='cum_21_days_projected')
    fig8.update_traces(name='Projected Total 21 days after Vaccination', showlegend=True, line_color='pink', hovertemplate='%{y:,.0f}', visible="legendonly")
    
    fig.add_trace(fig2.data[0])
    fig.add_trace(fig3.data[0])
    fig.add_trace(fig4.data[0])
    fig.add_trace(fig5.data[0])
    fig.add_trace(fig6.data[0])
    fig.add_trace(fig7.data[0])
    fig.add_trace(fig8.data[0])

    
    fig.update_layout(yaxis=dict(tickformat=',0.f', showgrid=False),
                    xaxis=dict(showgrid=False),
                    hovermode='x unified',
                    showlegend= True,
                    legend={
                            "orientation": "h",
                            "xanchor": "center",
                            'y': 1.2,
                            'x': 0.5,
                            'font': dict(size=9)
                            },
                    shapes=[
                            dict(type= 'line',  layer='below',
                                yref='y',
                                y0=13.4e6, y1=13.4e6,
                                xref='x', 
                                x0=datetime.strptime("8 December, 2020", "%d %B, %Y"),
                                x1=df[x].max() + timedelta(days=20),
                                line=dict(dash="dot", color=colors['lloyds'])),
                            dict(type= 'line', layer='below',
                                yref='y',
                                y0=29.6e6, y1=29.6e6,
                                xref='x', 
                                x0=datetime.strptime("8 December, 2020", "%d %B, %Y"),
                                x1=df[x].max() + timedelta(days=20),
                                line=dict(dash="dot", color=colors['lloyds']))
                            ])

    fig.add_annotation(x=datetime.strptime("15 February, 2021", "%d %B, %Y"), yref='y', y=13.4e6, ax=65, ay=65,
            text="Top 4 Priority<br>Groups Target", xanchor="left", xshift=5, showarrow=True, arrowhead=1, arrowsize = 2)
    fig.add_annotation(x=datetime.strptime("31 March, 2021", "%d %B, %Y"), yref='y', y=29.6e6, ax=60, ay=25,
            text="All Priority<br>Groups Target", xanchor="left", xshift=5, showarrow=True, arrowhead=1, arrowsize = 2)

    fig.update_yaxes(domain=(0, 0.9))
    fig.update_xaxes(range=[datetime.strptime("8 December, 2020", "%d %B, %Y"), 
                            df[x].max() + timedelta(days=20)], 
                            scaleratio = 0.1)

    return fig


def make_bar(df, x_title, y_title, x, y, rolling_avg):
    fig=px.bar(df, x=x, y=y,
             labels={x: x_title, y: y_title})
    
    fig.update_traces(marker_color=colors['maincolor'], hovertemplate='Daily: %{y:,.0f}', howlegend=True)

    fig2 = px.line(df, x=x, y='daily_rolling_average', line_shape='spline')
    fig2.update_traces(name=f'{rolling_avg} day rolling average', showlegend=True, line_color='#000000', hovertemplate='%{y:,.0f}')

    fig.add_trace(fig2.data[0])

    fig.update_layout(yaxis=dict(tickformat=',0.f', showgrid=False),
                      xaxis=dict(showgrid=False),
                     hovermode='x unified',
                     showlegend= True,
                     legend={
                            "orientation": "h",
                            "xanchor": "center",
                            'y': 1.2,
                            'x': 0.5,
                            'font': dict(size=9)
                            },)

    fig.update_xaxes(range=[datetime.strptime("11 January, 2021", "%d %B, %Y") + timedelta(hours=-12), 
                            df[x].max() + timedelta(days=-90, hours=12)], 
                            scaleratio = 0.1)
    return fig


def make_gauge(num_vax, previous_vax):
    
    pop = 66.65e6
    group_pops = [0, 1, 3, 2.5, 2.2, 3.3, 1.4, 3.3, 3.8, 4.4, 4.7, 37.05]
    group_pops = [x*1e6 for x in group_pops]
    group_prop = np.cumsum(group_pops)
    groups = ["Care Homes", "80+", "Frontline Health Workers", "75+", "70+", 
                   "Clinically Vulnerable", "65+",
                   "60+", "55+", "50+", "Everyone Else", "Entire Population"]
    colours = ["012a4a","013a63","01497c","014f86","2a6f97","2c7da0","468faf","61a5c2","89c2d9","a9d6e5",
              "c7e2eb"]
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
    fig = px.line(df, x=x, y=y, line_shape='spline',
                  labels={x: x_title,
                          y: y_title})

    fig.update_traces(line_color=colors['maincolor'], hovertemplate='%{y}')

    fig.update_layout(yaxis=dict(tickformat=',.0f'),
                      xaxis=dict(showgrid=False),
                      font=dict(family="IBM Plex Sans", size=12, color="#000000"),
                      hovermode="x unified")

    return fig


def make_daily_plot(df, x, y, x_title, y_title):
    fig = px.bar(df, x=x, y=y, labels={x: x_title, y: y_title})

    fig.update_traces(name=y_title, showlegend=True, marker_color=colors['maincolor'])

    fig2 = px.line(df, x=x, y='daily_rolling_average', line_shape='spline')
    fig2.update_traces(name='7 day rolling average', showlegend=True, line_color='#000000')

    fig.add_trace(fig2.data[0])

    fig.update_layout(yaxis=dict(tickformat=',.0f'),
                      font=dict(family="IBM Plex Sans", size=12, color="#000000"), showlegend=True,
                      legend=dict(x=0.99, y=0.9, xanchor='right'),
                      hovermode="x unified")

    fig.update_traces(hovertemplate='%{y}')

    return fig

