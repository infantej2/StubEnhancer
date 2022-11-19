import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__)

layout = html.Div(children=[
    html.Div(children=[
        dcc.Link(html.Button("Home", className="button"), href="/", refresh=True),
        dcc.Link(html.Button("by salary", className="button"), href="/salary", refresh=True,),
        dcc.Link(html.Button("by field", className="button-disabled", disabled=True), href="/field", refresh=True),
        dcc.Link(html.Button("prediction", className="button"), href="/prediction", refresh=True),
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