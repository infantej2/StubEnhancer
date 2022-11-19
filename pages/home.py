import dash
from dash import html, dcc

dash.register_page(__name__, path='/')

layout = html.Div(children=[
    html.H1(children='Stub Enhancer'),
    html.Div(children="Welcome to Stub Enhancer! We aim to help you enhance your pay "
                        "stub by providing data abstractions based on data from ALIS. "
                        "Our goal is to aid Albertans in their career and education "
                        "decisions."),
    html.Br(),
    html.Div(children=[
        dcc.Link(html.Button("Get Started!", className="button"), href="/salary", refresh=True),
    ]),
    html.Br(),
    html.Div(children="The ideal income, according to a study by purdue university "
                        "is $127l. They also note the emotinoal wellbeing threshold is "
                        "78K-$97K."),

])