import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html, callback
import numpy as np
import pandas as pd
import plotly.graph_objects as go

dash.register_page(__name__)

# -------------------------------------------------------------------------------------------------------------


# Dataframe cleaning and prep
df = pd.read_csv('./abSchool.csv')
df = df.dropna()
df = df.drop(columns=['Cohort Size'])
df['Median Income'] = df['Median Income'].str.replace('[$,]', '', regex=True).astype(int)
dflist = df.loc[:,"Field of Study (2-digit CIP code)"]
list = dflist.to_numpy()
list = np.unique(list)

def_list = ["01.00. Agriculture, general",
"01.01. Agricultural business and management",
"01.02. Agricultural mechanization",
"01.05. Agricultural and domestic animal services",
"01.06. Applied horticulture/horticultural business services"]


# -------------------------------------------------------------------------------------------------------------


layout = html.Div(children=[
    html.Header(children=[
        dcc.Link(html.Button("Home", className="button", style={"margin-left":"20px"}), href="/", refresh=True),
        dcc.Link(html.Button("by salary", className="button"), href="/salary", refresh=True,),
        dcc.Link(html.Button("by field", className="button-disabled", disabled=True), href="/field", refresh=True),
        dcc.Link(html.Button("prediction", className="button"), href="/prediction", refresh=True),
    ], className="header"),
    html.H1(children='By Field', style={'text-align':'center'}),
    html.P([
        html.H2("Average Median Income by Field of Study --- Dropdown 2-digit CIP",
        style={'text-align': 'center', 'padding-top':'100px'}),
        html.Div(children=[
        html.H5("Add fields: "),
        html.Div([
            dcc.Dropdown(list, id="my-input1", value="01. Agriculture, agriculture operations and related sciences", multi=False, style={'width':'90%', 'margin-left':'30px'})
        ], style={'width':'100%', 'display':'flex', 'align-items':'center', 'justify-content':'center'}),
        ], style={'text-align': 'center'}),
        html.Div(id='my-output1')
        ])
])


# --------------------------------------------------------------------------------------------------------------------------------------

# Explore 2-digit CIP update and callback
@callback(
    Output(component_id='my-output1', component_property='children'),
    Input(component_id='my-input1', component_property='value')
)

def update_output2(input2):

    '''
    temp_df1 = (df.loc[df['Field of Study (4-digit CIP code)'].str.lower().str.contains(input2[0-2])])
    df1 = temp_df1.groupby(['Field of Study (4-digit CIP code)']).mean(numeric_only=True).astype(int).sort_values('Median Income', ascending=True)
    df1.reset_index(inplace=True)
    '''

    temp_df2 = (df.loc[df['Field of Study (2-digit CIP code)'] == input2])
    df2 = temp_df2.groupby(['Credential']).mean(numeric_only=True).astype(int).sort_values('Median Income', ascending=True)
    df2.reset_index(inplace=True)

    trace = go.Bar(
        x=df2["Credential"],
        y=df2["Median Income"],
        text = df2["Median Income"],
        textposition="inside",
        marker=dict(color='#024B7A'),
        marker_line=dict(width=1, color='black'),
        width=0.5,
        hovertemplate='<extra></extra><br>Credential: %{x} <br>Average Median Income: %{y}'
    )

    layout = go.Layout(
        margin=go.layout.Margin(
        l=150,   # left margin
        r=150,   # right margin
        b=100,   # bottom margin
        t=50    # top margin
        ), 
        height = 700,
        title_x = 0.5,
        xaxis_title="Credential",
        yaxis_title="Median Income ($)",
        bargap=0
    )

    barChart = dcc.Graph(figure=go.Figure(data=trace, layout=layout))
    return barChart


# --------------------------------------------------------------------------------------------------------------------------------------