import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go

app = dash.Dash(
	external_stylesheets=[dbc.themes.BOOTSTRAP],
	use_pages=True,
	suppress_callback_exceptions=True,
	pages_folder=''
)

server = app.server

app.layout = html.Div([
	dash.page_container
])

if __name__ == '__main__':
	app.run_server(
		debug=True
	)