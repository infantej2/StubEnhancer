import dash
from dash import html, dcc

dash.register_page(__name__)

layout = html.Div(children=[
    html.Header(children=[
        dcc.Link(html.Button("Home", className="button"), href="/", refresh=True),
        dcc.Link(html.Button("by salary", className="button"), href="/salary", refresh=True,),
        dcc.Link(html.Button("by field", className="button"), href="/field", refresh=True),
        dcc.Link(html.Button("prediction", className="button-disabled", disabled=True), href="/prediction", refresh=True),
    ], className="header"),
    html.H1(children='Prediction'),

    html.Div(children='''
        This is our Prediction Page
    '''),

])