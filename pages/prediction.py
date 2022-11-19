import dash
from dash import html, dcc

dash.register_page(__name__)

layout = html.Div(children=[
    html.Div(children=[
        dcc.Link(html.Button("by salary"), href="/salary", refresh=True,),
        dcc.Link(html.Button("by field"), href="/field", refresh=True),
        dcc.Link(html.Button("prediction"), href="/prediction", refresh=True),
    ]),
    html.H1(children='Prediction'),

    html.Div(children='''
        This is our Archive page content.
    '''),

])