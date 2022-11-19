import dash
from dash import html, dcc

dash.register_page(__name__)

layout = html.Div(children=[
    html.Div(children=[
        dcc.Link(html.Button("Home", className="button"), href="/", refresh=True),
        dcc.Link(html.Button("by salary", className="button-disabled", disabled=True), href="/salary", refresh=True,),
        dcc.Link(html.Button("by field", className="button"), href="/field", refresh=True),
        dcc.Link(html.Button("prediction", className="button"), href="/prediction", refresh=True),
    ]),
    html.H1(children='By Salary'),
    html.Div(children='''
        This is our Salary Page
    '''),
    
])