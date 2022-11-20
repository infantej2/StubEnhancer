from .shared import generate_header

import dash
from dash import html, dcc

dash.register_page(__name__)

layout = html.Div(children=[
    generate_header(__name__),
    html.H1(children='By Salary'),
    html.Div(children='''
        This is our Salary Page
    '''),
    
])