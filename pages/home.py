import dash
from dash import Dash, Input, Output, dcc, html, callback
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .shared import generate_navbar

derived_df = pd.read_csv('./derived_data.csv')

dash.register_page(__name__, path='/')

# -------------------------------------------------------------------------------------------------------------


def jobs_happiness_scatterplot():
    layout = go.Layout(
        margin=go.layout.Margin(
            l=100,   # left margin
            r=50,   # right margin
            b=100,   # bottom margin
            t=100    # top margin
        ),
        height=700,
        title_x=0.5,
        title='Average Job Salaries (CAD) Above the Happiness Threshold<br>(at 10th Year Salary) (2005-2014)',
        xaxis={
            'title':{
                'text': "Job Order as a Function of Salary (Ascending)",
                'standoff': 30
            },
        },
        yaxis_title="Average Income Ten Years After Graduation (CAD)",
        bargap=0,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
    )

    display_df = derived_df.loc[derived_df['Credential']
                                == 'Overall (All Graduates)']
    display_df = display_df.dropna(axis=0)
    display_df = display_df.sort_values(
        'Average Income Ten Years After Graduation', ascending=True).reset_index()

    fig = px.scatter(
        display_df,
        x=display_df.index,
        y='Average Income Ten Years After Graduation'
    )

    fig.layout = layout

    fig.update_traces(marker = {'color': '#D84FD2'})
    fig.update_xaxes(visible=True, showticklabels=False, showgrid=False, zeroline=False, gridcolor='#cc99ff')
    fig.update_yaxes(gridcolor='#666666')

    fig.add_hrect(
        78000, 97000,
        annotation_text='Emotional Well-being', annotation_position='top left',
        annotation=dict(
            font_size=20, font_family='Times New Roman', font_color='white'),
        fillcolor="#3BB143", opacity=0.25, line_width=0
    )

    fig.add_hline(
        123000,
        annotation_text='Ideal Income',
        annotation_position='top left',
        annotation=dict(
            font_size=20, font_family='Times New Roman', font_color='white'),
        line_color='#3BB143'
    )

    fig.add_vline(
        int(len(display_df.index) / 2),
        line_width=1,
        line_dash='dash',
        annotation_text='50th Percentile',
        annotation_position='bottom',
        annotation=dict(
            font_size=16, font_family='Times New Roman', font_color='white'),
        line_color='white'
    )

    fig.add_vline(
        int(len(display_df.index) * 0.80),
        line_width=1,
        line_dash='dash',
        annotation_text='80th Percentile',
        annotation_position='bottom',
        annotation=dict(
            font_size=16, font_family='Times New Roman', font_color='white'),
        line_color='white'
    )

    fig.add_vline(
        int(len(display_df.index) * 0.95),
        line_width=1,
        line_dash='dash',
        annotation_text='95th Percentile',
        annotation_position='bottom',
        annotation=dict(
            font_size=16, font_family='Times New Roman', font_color='white'),
        line_color='white'
    )

    fig.update_traces(
        hovertemplate='Average Income Ten Years After Graduation: %{y}'
    )

    fig.update_layout(yaxis_tickprefix = '$',
                        font=dict(
                            size=14
                        ))

    return dcc.Graph(
        figure=fig,
        config={'displayModeBar': False}
    )

# -------------------------------------------------------------------------------------------------------------


def certification_salaries_barchart():
    layout = go.Layout(
        margin=go.layout.Margin(
            l=150,   # left margin
            r=150,   # right margin
            b=100,   # bottom margin
            t=100    # top margin
        ),
        height=700,
        title_x=0.5,
        title='Mean Incomes by Certification Type',
        xaxis_title="Mean Income (CAD)", #xaxis_title="Credential",
        yaxis_title="Credential", # yaxis_title="Mean Income (CAD)",
        bargap=0,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        yaxis_showticklabels=False
    )

    figure = go.Figure(
        layout=layout
    )

    dataframe = derived_df.loc[derived_df['Field of Study (CIP code)'].str.contains(
        '00. Total')]
    dataframe = dataframe.loc[dataframe['Credential']
                              != 'Overall (All Graduates)']
    dataframe = dataframe.reset_index()

    # The map of colors for each type of credential
    # NOTE: The credential types will be pulled in the order specified within the list.
    color_map = {
        'Certificate': '#E91F63',
        'Diploma': '#9D27B0',
        'Bachelor\'s degree': '#262AAF',
        'Professional bachelor\'s degree': '#266BAF',
        'Bachelor\'s degree + certificate/diploma': '#57ACDD',
        'Master\'s degree': '#58DCBD',
        'Doctoral Degree': '#61C688',
    }

    # Add each bar by certification type
    for certificate_name in color_map:
        # Get the new dataframe for this specific bar (by its credential name)
        bar_df = dataframe[dataframe["Credential"] == certificate_name]

        # Construct the new bar trace
        new_trace = go.Bar(
            x=bar_df['Average Income Ten Years After Graduation'], # x=bar_df["Credential"],
            y=bar_df["Credential"], # y=bar_df['Average Income Ten Years After Graduation'],
            text=bar_df['Average Income Ten Years After Graduation'],
            textposition="inside",
            texttemplate='%{y} %{x}',
            orientation='h', #orientation='v',
            #textangle=-90,
            marker=dict(color='#024B7A'),
            marker_line=dict(width=1, color='black'),
            marker_color=color_map[certificate_name],
            width=0.85,
            hovertemplate='<extra></extra><br>Credential: %{x} <br>Average Median Income: %{y}',
            name=certificate_name,
            showlegend=False
        )

        # Add the new bar trace into the overall figure
        figure.add_traces(new_trace)
        figure.update_xaxes(gridcolor='#666666')
        figure.update_layout(xaxis_tickprefix = '$',
                        font=dict(
                            size=14
                        ))

    barChart = dcc.Graph(
        figure=figure,
        config={'displayModeBar': False}
    )

    return barChart

# -------------------------------------------------------------------------------------------------------------


def top_vs_bottom_5_barchart():
    layout = go.Layout(
        margin=go.layout.Margin(
            l=150,   # left margin
            r=150,   # right margin
            b=50,   # bottom margin
            t=100    # top margin
        ),
        height=700,
        title_x=0.5,
        title='Top 5 vs Bottom 5 Jobs by 10th Year Salary (CAD)',
        xaxis_title="Mean Income 10 Years After Graduation (CAD)", # xaxis_title="Top 5 / Bottom 5 Jobs",
        yaxis_title="Top 5 / Bottom 5 Jobs", # yaxis_title="Mean Income 10 Years After Graduation (CAD)",
        bargap=0,
        uniformtext_minsize=10,
        uniformtext_mode='show',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
    )

    dataframe = derived_df.loc[derived_df['Credential']
                               == 'Overall (All Graduates)']
    dataframe = dataframe.dropna(axis=0)
    dataframe = dataframe.sort_values(
        'Average Income Ten Years After Graduation', ascending=False)

    # Remove the 4 digit code from the start of the FoS string
    dataframe['Field of Study (CIP code)'] = dataframe['Field of Study (CIP code)'].str.replace(
        '[0-9]{2}.[0-9]{2} ', '', regex=True)

    top = dataframe.head(5)
    bottom = dataframe.tail(5)
    dataframe = pd.concat([top, bottom])

    figure = px.bar(
        data_frame=dataframe,
        x='Average Income Ten Years After Graduation', # x='Field of Study (CIP code)',
        y='Field of Study (CIP code)', # y='Average Income Ten Years After Graduation',
        text='Average Income Ten Years After Graduation',
        #color=['Top 5', 'Top 5', 'Top 5', 'Top 5', 'Top 5', 'Bottom 5', 'Bottom 5', 'Bottom 5', 'Bottom 5', 'Bottom 5'],
        # color_discrete_map={
        #    'Top 5': '#D84FD2',
        #    'Bottom 5': '#8B4FD8'
        # },

        orientation='h'
    )

    figure.data[0].marker.color = ('#D84FD2', '#D84FD2', '#D84FD2', '#D84FD2', '#D84FD2', '#8B4FD8', '#8B4FD8', '#8B4FD8', '#8B4FD8', '#8B4FD8')

    # Apply the layout described at the top
    figure.layout = layout

    # Manually add a legend to the graph
    figure.update_traces(showlegend=False).add_traces(
        [
            go.Bar(name=m[0], y=[figure.data[0].x[0]],
                   marker_color=m[1], showlegend=True, orientation='h')
            for m in [('Bottom 5', '#8B4FD8'), ('Top 5', '#D84FD2')]
        ]
    )

    # Display the text and value as a string inside the bar, with vertical orientation
    figure.update_traces(
        texttemplate='%{y} %{x}', #texttemplate='%{x} %{y}',
        textposition=['inside', 'inside', 'inside', 'inside', 'inside',
                      'outside', 'outside', 'outside', 'outside', 'outside'],
        #orientation='v',
        #textangle=-90,
        hovertemplate='Field of Study: %{y}<br>Average Income Ten Years After Graduation: %{x}', # hovertemplate='Field of Study: %{x}<br>Average Income Ten Years After Graduation: %{y}',
        width = 0.85
    )

    # Remove field labels
    figure.update_yaxes(visible=False, showticklabels=False) # figure.update_xaxes(visible=False, showticklabels=False)
    figure.update_xaxes(gridcolor='#666666')
    figure.update_layout(barmode='stack',
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis_tickprefix = '$',
                        font=dict(
                            size=14
                        ))

    barChart = dcc.Graph(
        figure=figure,
        config={'displayModeBar': False}
    )

    return barChart

# -------------------------------------------------------------------------------------------------------------


layout = html.Div(className="body", children=[
    generate_navbar(__name__),
    html.Div(className="home-wrapper", children=[
        html.Div(className="home-one", children=[
            html.Div(children=["Welcome to ", html.Span("Stub Enhancer", style={'color':'#D84FD2'}), "! We aim to help you enhance your pay "
                     "stub by providing data abstractions based on data from ALIS. "
                     "Our goal is to aid Albertans in their career and education "
                     "decisions."], style={"color": "white", "fontSize":"18px", "padding":"20px"}),
            html.Br(),
            html.Div(children=[
                dcc.Link(html.Button("Get Started!", className="button-start"),
                 href="/salary", refresh=False),
            ], style={"paddingLeft":"20px", "paddingRight":"20px"}),
            html.Br(),
             html.Div(children=[
                html.Span("Life evaluation", style={'color':'#D84FD2'}), ', really life satisfaction, is an overall assessment of how '
                "one is doing and is likely more infuenced by higher goals and comparisons "
                'to others. ', html.Span("Emotional well-being", style={'color':'#D84FD2'}), ", or feelings, is about one's day-to-day "
                "emotions, such as feeling happy, excited, or sad and angry."
            ], style={"color": "white", "fontSize":"18px", "padding":"20px"}
            ),
            html.Br(),
            html.Div(children=[
                "The ", html.Span("ideal income for life evaluation", style={'color':'#D84FD2'}),", according to a ",
                html.A("study by Purdue University",
                       href='https://www.purdue.edu/newsroom/releases/2018/Q1/money-only-buys-happiness-for-a-certain-amount.html'),
                " is ", html.Span("$127K.", style={'color':'#D84FD2'}), " They also note the ", html.Span("emotional wellbeing", style={'color':'#D84FD2'}), " threshold is ", html.Span("78K-$97K.", style={'color':'#D84FD2'})
            ], style={"color": "white", "fontSize":"18px", "padding":"20px"}
            ),
        ]),
        html.Div(className="home-two", children=[
            dbc.Tabs([
                dbc.Tab(jobs_happiness_scatterplot(),
                        label="Happiness Threshold",
                        label_style={"color": "#D84FD2"}),
                dbc.Tab(certification_salaries_barchart(),
                        label="Certification",
                        label_style={"color": "#D84FD2"}),
                dbc.Tab(top_vs_bottom_5_barchart(),
                        label="Top 5 vs Bottom 5",
                        label_style={"color": "#D84FD2"}),
            ], )
        ]),
    ],),
])
