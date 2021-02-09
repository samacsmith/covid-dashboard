import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime, timedelta
from dash.dependencies import Input, Output
from get_data import get_covid_data, make_cum_vaccine_plot, make_bar, make_gauge

def update_data():
    df, last_update, rolling_avg, end_date = get_covid_data()

    cum_first_dose = make_cum_vaccine_plot(df, end_date)
    daily_dose = make_bar(df.loc[(df['date'] > datetime.strptime("10 January, 2021", "%d %B, %Y"))], 'Date (reported)', 'Daily Number of Doses (First and Second)', 'date', 'daily_total_doses', rolling_avg)
    gauge_chart = make_gauge(df.set_index('date').loc[end_date, 'cum_first_dose'],
                        df.set_index('date').loc[end_date+timedelta(days=-1), 'cum_first_dose'])

    return last_update, cum_first_dose, daily_dose, gauge_chart
    

def serve_layout():

    global last_update
    global cum_first_dose
    global daily_dose
    global gauge_chart

    last_update, cum_first_dose, daily_dose, gauge_chart = update_data()
    
    return html.Div(id='whole-page', className='container', children=[

    html.Div(id='header', className='row', children=[

        html.Div(id='titles', className='nine columns', children=[

            html.H3(children='Coronavirus in the UK')            
        ]),

        html.Div(id='updates', className='three columns', children=[
            html.H5(children=f'Data up to: {last_update}', style={'text-align': 'right', 'font-family': "Roboto Mono", 'font-size': '0.8rem'}),
            html.H5(children=f'Page last updated: {datetime.now().strftime("%a %d %b %H:%M")}', style={'text-align': 'right', 'font-family': "Roboto Mono", 'font-size': '0.8rem'})            
        ])    
    ]),

    html.Div(className='tabs row', children=[

        dcc.Tabs(id='tabs', className='custom-tabs', value='tab-1', children=[

            dcc.Tab(id='tab-1', label='Vaccinations', value='tab-1', className="custom-tab",
                    selected_className="custom-tab--selected"),
            dcc.Tab(id='tab-2', label='Cases', value='tab-2', className="custom-tab",
                    selected_className="custom-tab--selected"),
            dcc.Tab(id='tab-3', label='Deaths', value='tab-3', className="custom-tab",
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
        return html.Div(id='whole-tab', children=[
            html.H4('Some more detailed stuff about cases....')
        ])

    elif tab == 'tab-3':
        return html.Div(id='whole-tab', children=[
            html.H4('Some more detailed stuff about deaths....')
        ])


if __name__ == '__main__':
    app.run_server(debug=True)


'''
To Do:
1) User defined vaccination efficacy (for number immune and cases)
2) Take into account second doses
3) Headline figures/dates
4) log plot with quadratic
5) Update number in each priority group and stop first doses after all done
6) bar chart for total doses
'''
