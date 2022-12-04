from .shared import generate_header, generate_navbar

import dash
from dash import Dash, Input, Output, dcc, html, callback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math

dash.register_page(__name__)

# -------------------------------------------------------------------------------------------------------------

credential_map = {
    'Certificate': 'red',
    'Diploma': 'green',
    'Bachelor\'s degree': 'blue',
    'Professional bachelor\'s degree': 'yellow',
    'Bachelor\'s degree + certificate/diploma': 'grey',
    'Master\'s degree': 'orange',
    'Doctoral Degree': 'teal',
}

credential_list = list(credential_map.keys())

derived_df = pd.read_csv('./derived_data.csv')

min_max_df = derived_df.loc[derived_df['Credential'] != 'Overall (All Graduates)']
min_salary = int(math.floor(min_max_df['Average Income Ten Years After Graduation'].min() / 1000.0) * 1000.0) # Floor to closest multiple of 1000 (42,950 -> 42,000)
max_salary = int(math.ceil(min_max_df['Average Income Ten Years After Graduation'].max() / 1000.0) * 1000.0) # Ceil to closest multiple of 1000 (55,422 -> 56,000)

# -------------------------------------------------------------------------------------------------------------

layout = html.Div(className="body", children=[
    generate_navbar(__name__),
    # generate_header(__name__),
    html.Div(
        html.H1(className="title", children='By Salary'),
    ),
    html.Div(children=[
        html.Div(children=[
            '''
            Salary Range
            '''
        ], style={"color": "white", "fontSize": "25px", "marginRight": "50px"}),
        html.Div(children=[
            html.Div(children=[
                html.Div(id='Salary-Range-Min',
                         style={"color": "white", "textAlign": "left"}),
                html.Div(id='Salary-Range-Max',
                         style={"color": "white", "textAlign": "right"}),
            ], style={"display": "flex", "justify-content": "space-between"}),
            dcc.RangeSlider(min_salary, max_salary, step=1000, value=[60000, 80000],
            marks=None, id='Salary-Range-Slider', tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"width": "100%"}),
    ], style={"display": "flex", "width": "70%", "margin": "auto", "paddingTop": "40px"}),
    html.Div(children=[
        html.Div(children=[
            '''
            Max Items
            '''
        ], style={"color": "white", "fontSize": "25px", "marginRight": "11%"}),
        html.Div(children=[
            dcc.Slider(3, 25, 1, value=10, marks=None, id='Job-Display-Max',
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"width": "100%", "paddingTop": "15px"}),
    ], style={"display": "flex", "width": "70%", "margin": "auto", "paddingTop": "20px"}),
    html.Div(children=[
        dcc.Checklist(
            id='Credential-Checklist',
            options=credential_list,
            value=credential_list,
            inline=True,
            labelStyle={'display': 'block'},
            # TODO: I couldn't figure out how to center these... 
            style={"color": "white", "fontSize": "20px", "padding": "20px", "width": "100%", "margin": "auto", "text-indent": "2em"},
            className="salary-two"
        ),
        html.Div(id='Jobs-By-Salary-Barchart', className="salary-one"),
    ], className="salary-wrapper", style={"width": "100%", "margin": "auto", "display": "table"}),
])

# -------------------------------------------------------------------------------------------------------------


@callback(
    Output(component_id='Salary-Range-Min', component_property='children'),
    Input(component_id='Salary-Range-Slider', component_property='value')
)
def update_min_salary_text(salary_range):
    #return f'Min: ${salary_range[0]} CAD'
    return f'Min: ${min_salary:,} CAD'


@callback(
    Output(component_id='Salary-Range-Max', component_property='children'),
    Input(component_id='Salary-Range-Slider', component_property='value')
)
def update_max_salary_text(salary_range):
    #return f'Max: ${salary_range[1]} CAD'
    return f'Max: ${max_salary:,} CAD'

# -------------------------------------------------------------------------------------------------------------


@callback(
    Output(component_id='Jobs-By-Salary-Barchart',
           component_property='children'),
    Input(component_id='Credential-Checklist', component_property='value'),
    Input(component_id='Salary-Range-Slider', component_property='value'),
    Input(component_id='Job-Display-Max', component_property='value')
)
def update_jobs_by_salary_graph(credentials, salary_range, jobs_display_max):
    layout = go.Layout(
        margin=go.layout.Margin(
            l=150,   # left margin
            r=150,   # right margin
            b=100,   # bottom margin
            t=50    # top margin
        ),
        height=700,
        title_x=0.5,
        title='Fields with an Average Income Within the Selected Salary Range',
        xaxis_title="Mean Income (CAD)", # xaxis_title="Field & Certification",
        yaxis_title="Field & Certification", # yaxis_title="Mean Income (CAD)",
        # bargap=0,
        barmode='group',
        uniformtext_minsize=10,
        uniformtext_mode='show',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        showlegend=False
    )

    figure = go.Figure(
        layout=layout
    )

    salary_min = salary_range[0]
    salary_max = salary_range[1]

    dataframe = derived_df.loc[derived_df['Field of Study (CIP code)'].str.contains(
        '00. Total') == False]
    #dataframe = dataframe.loc[dataframe['Credential'] != 'Overall (All Graduates)']

    if (len(credentials) > 0) and (dataframe.empty == False):
        dataframe = dataframe.loc[dataframe['Credential'].str.contains(
        '|'.join(credentials))]

        dataframe = dataframe[dataframe['Average Income Ten Years After Graduation'] >= salary_min]
        dataframe = dataframe[dataframe['Average Income Ten Years After Graduation'] <= salary_max]
        dataframe = dataframe.sort_values(
            'Average Income Ten Years After Graduation', ascending=False).head(jobs_display_max).reset_index()

        # Remove the 2 digit code from the start of the FoS string
        dataframe['Field of Study (CIP code)'] = dataframe['Field of Study (CIP code)'].str.replace(
            '[0-9]{2}. ', '', regex=True)

        # Update Field of Study to include credential type (or else they'll group/overlap)
        for index, row in dataframe.iterrows():
            dataframe.loc[index, 'Field of Study (CIP code)'] = row['Field of Study (CIP code)'] + \
                ' (' + row['Credential'] + ')'
            dataframe.loc[index, 'Color'] = credential_map[row['Credential']]

        for credential in credentials:
            bar_df = dataframe.loc[dataframe['Credential'] == credential]

            figure.add_trace(
                go.Bar(
                    x=bar_df['Average Income Ten Years After Graduation'], # x=bar_df['Field of Study (CIP code)'],
                    y=bar_df['Field of Study (CIP code)'], # y=bar_df['Average Income Ten Years After Graduation'],
                    #text=bar_df['Average Income Ten Years After Graduation'],
                    texttemplate='%{y} %{x}', # texttemplate='%{x} %{y}',
                    textposition="inside",
                    marker=dict(color='#024B7A'),
                    marker_line=dict(width=1, color='black'),
                    marker_color=credential_map[credential],
                    width=0.8, # width=0.5,
                    hovertemplate='<extra></extra><br>Credential: %{y} <br>Average Median Income: %{x}', # hovertemplate='<extra></extra><br>Credential: %{x} <br>Average Median Income: %{y}',
                    name=credential,
                    showlegend=False,

                    orientation='h'
                )
            )

    # Remove field labels
    figure.update_yaxes(visible=True, showticklabels=False) # figure.update_xaxes(visible=True, showticklabels=False)
    figure.update_xaxes(visible=True, showticklabels=True, showgrid=True, zeroline=True)

    #figure.update_traces(orientation='v', textangle=-90)

    # Remove the grouping
    figure.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'}) # figure.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})

    chart = dcc.Graph(
        figure=figure,
        config={'displayModeBar': False}
    )

    return chart
