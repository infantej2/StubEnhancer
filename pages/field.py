import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__)

layout = html.Div(children=[
    html.Div(children=[
        dcc.Link(html.Button("by salary"), href="/salary", refresh=True,),
        dcc.Link(html.Button("by field"), href="/field", refresh=True),
        dcc.Link(html.Button("prediction"), href="/prediction", refresh=True),
    ]),
    html.H1(children='By Field'),
	html.Div([
        "Select a city: ",
        dcc.RadioItems(['New York City', 'Montreal','San Francisco'],
        'Montreal',
        id='analytics-input')
    ]),
	html.Br(),
    html.Div(id='analytics-output'),
])


@callback(
    Output(component_id='analytics-output', component_property='children'),
    Input(component_id='analytics-input', component_property='value')
)
def update_city_selected(input_value):
    return f'You selected: {input_value}'