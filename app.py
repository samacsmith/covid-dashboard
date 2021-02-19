import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime, timedelta
from dash.dependencies import Input, Output
from get_data import get_covid_data, make_cum_vaccine_plot, make_bar, make_gauge, make_7da_plot, make_indexed_plot

def update_data():
    vacc_df, last_update, rolling_avg, end_date, recent_cases_df, \
            recent_deaths_df, recent_admissions_df, admissions_by_age_df, \
            ad_age_groups, cases_by_age_df, cases_age_groups, \
            deaths_by_age_df, deaths_age_groups = get_covid_data()

    vacc_df = vacc_df.rename(columns={'daily_first_dose': 'First Dose', 'daily_second_dose': 'Second Dose'})

    cum_first_dose = make_cum_vaccine_plot(vacc_df, end_date)
    daily_dose = make_bar(vacc_df.loc[(vacc_df['date'] > datetime.strptime("10 January, 2021", "%d %B, %Y"))], 'Date (reported)', 'Daily Number of Doses', 'date', ["First Dose", "Second Dose"], rolling_avg)
    gauge_chart = make_gauge(vacc_df.set_index('date').loc[end_date, 'cum_first_dose'],
                        vacc_df.set_index('date').loc[end_date+timedelta(days=-1), 'cum_first_dose'])
    fitted_cases = make_7da_plot(recent_cases_df, log=True, metric='Cases')
    fitted_deaths = make_7da_plot(recent_deaths_df, log=True, metric='Deaths')
    fitted_admissions = make_7da_plot(recent_admissions_df, log=True, metric='Admissions')
    age_admissions = make_indexed_plot(admissions_by_age_df, [x.replace('indexed', ' ').replace('_', ' ') for x in ad_age_groups.keys()], log=False, metric='Admissions')
    age_cases = make_indexed_plot(cases_by_age_df, [x.replace('indexed', ' ').replace('_', ' ') for x in cases_age_groups.keys()], log=False, metric='Cases')
    age_deaths = make_indexed_plot(deaths_by_age_df, [x.replace('indexed', ' ').replace('_', ' ') for x in deaths_age_groups.keys()], log=False, metric='Deaths')

    return last_update, cum_first_dose, daily_dose, gauge_chart, fitted_cases, fitted_deaths, fitted_admissions, age_admissions, age_cases, age_deaths


def serve_layout():

    global last_update
    global cum_first_dose
    global daily_dose
    global gauge_chart
    global fitted_cases
    global fitted_deaths
    global fitted_admissions
    global age_admissions
    global age_cases
    global age_deaths

    last_update, cum_first_dose, daily_dose, gauge_chart, fitted_cases, fitted_deaths, fitted_admissions, age_admissions, age_cases, age_deaths = update_data()


    
    return html.Div(id='whole-page', className='container', children=[

    html.Div(id='header', className='row', children=[

        html.Div(id='titles', className='nine columns', children=[

            html.H3(children='Coronavirus in the UK')            
        ]),

        html.Div(id='updates', className='three columns', children=[
            html.H5(children=f'Data last updated: {last_update}', style={'text-align': 'right', 'font-family': "Roboto Mono", 'font-size': '0.8rem'}),
            html.H5(children=f'Page last updated: {datetime.now().strftime("%a %d %b %H:%M")}', style={'text-align': 'right', 'font-family': "Roboto Mono", 'font-size': '0.8rem'})            
        ])    
    ]),

    html.Div(className='tabs row', children=[

        dcc.Tabs(id='tabs', className='custom-tabs', value='tab-1', children=[

            dcc.Tab(id='tab-1', label='Vaccinations', value='tab-1', className="custom-tab",
                    selected_className="custom-tab--selected"),
            dcc.Tab(id='tab-2', label='Cases', value='tab-2', className="custom-tab",
                    selected_className="custom-tab--selected"),
            dcc.Tab(id='tab-3', label='Healthcare', value='tab-3', className="custom-tab",
                    selected_className="custom-tab--selected"),
            dcc.Tab(id='tab-4', label='Deaths', value='tab-4', className="custom-tab",
                    selected_className="custom-tab--selected"),

        ]),

    ]),

    html.Div(id='tabs-content')

])

app = dash.Dash(__name__)
server = app.server
app.config["suppress_callback_exceptions"] = True

app.title = 'COVID-19 Dashboard'

app.layout = serve_layout


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    
    if tab == 'tab-1':
        return html.Div(className='whole-tab', children=[
            html.Div(className='row flex-container', children=[
                html.Div(className='one-half column pretty-container', children=[
                            dcc.Graph(id='cum-vac-graph', className='plotly-graph', figure=cum_first_dose)
                    ]),
                html.Div(className='one-half column pretty-container', children=[
                            dcc.Graph(id='daily-vac-graph', className='plotly-graph', figure=daily_dose)
                    ])
                ]),
            html.Div(className='row flex-container', children=[
                html.Div(className='column pretty-container', children=[
                            dcc.Graph(id='gauge-graph', className='plotly-graph', figure=gauge_chart)
                    ])                
                ])
        ])
                        
    elif tab == 'tab-2':
        return html.Div(className='whole-tab', children=[
            html.Div(className='row flex-container', children=[
                html.Div(className='one-half column pretty-container', children=[
                            dcc.Graph(id='fitted-cases-graph', className='plotly-graph', figure=fitted_cases)
                    ]),
                html.Div(className='one-half column pretty-container', children=[
                            dcc.Graph(id='age-cases-graph', className='plotly-graph', figure=age_cases)
                    ])
                ])
        ])

    elif tab == 'tab-3':
        return html.Div(className='whole-tab', children=[
            html.Div(className='row flex-container', children=[
                html.Div(className='one-half column pretty-container', children=[
                            dcc.Graph(id='fitted-admissions-graph', className='plotly-graph', figure=fitted_admissions)
                    ]),
                html.Div(className='one-half column pretty-container', children=[
                            dcc.Graph(id='age-admissions-graph', className='plotly-graph', figure=age_admissions)
                    ])
                ])
        ])
    
    elif tab == 'tab-4':
        return html.Div(className='whole-tab', children=[
            html.Div(className='row flex-container', children=[
                html.Div(className='one-half column pretty-container', children=[
                            dcc.Graph(id='fitted-deaths-graph', className='plotly-graph', figure=fitted_deaths)
                    ]),
                html.Div(className='one-half column pretty-container', children=[
                            dcc.Graph(id='age-deaths-graph', className='plotly-graph', figure=age_deaths)
                    ])
                ])
        ])


if __name__ == '__main__':
    app.run_server(debug=True)


'''
To Do:
1) User defined vaccination efficacy (for number immune and cases)
2) Headline figures/dates
3) quadratic fit to cases, admissions, deaths -- when appropriate
4) % comparison for things (e.g. % >80 hospital admissions)
5) Update number in each priority group and stop first doses after all done
6) explainers for each graph
7) fix cumulative projections dropping
'''
