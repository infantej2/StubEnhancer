import dash
from dash import Dash, Input, Output, dcc, html, callback
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

derived_df = pd.read_csv('./derived_data.csv')

dash.register_page(__name__, path='/')

# -------------------------------------------------------------------------------------------------------------

def jobs_happiness_scatterplot():
    layout = go.Layout(
        margin=go.layout.Margin(
            l=150,   # left margin
            r=150,   # right margin
            b=100,   # bottom margin
            t=50    # top margin
        ), 
        height = 700,
        title_x = 0.5,
        title='Average Job Salaries (CAD) Above the Happiness Threshold (at 10th Year Salary) (2005-2014)',
        xaxis_title="Job Order as a Function of Salary (Ascending)",
        yaxis_title="Average Income Ten Years After Graduation (CAD)",
        bargap=0
    )

    display_df = derived_df.loc[derived_df['Credential'] == 'Overall (All Graduates)']
    display_df = display_df.dropna(axis=0)
    display_df = display_df.sort_values('Average Income Ten Years After Graduation', ascending = True).reset_index()

    fig = px.scatter(
        display_df,
        x = display_df.index,
        y = 'Average Income Ten Years After Graduation'
    )

    fig.layout = layout

    fig.add_hrect(
        78000, 97000,
        annotation_text='Emotional Well-being', annotation_position='top left',
        annotation=dict(font_size=20, font_family='Times New Roman', font_color='black'),
        fillcolor="green", opacity=0.25, line_width=0
    )

    fig.add_hline(
        123000,
        annotation_text='Ideal Income', 
        annotation_position='top left',
        annotation=dict(font_size=20, font_family='Times New Roman', font_color='black'),
    )

    fig.add_vline(
        int(len(display_df.index) / 2),
        line_width=1,
        line_dash='dash',
        annotation_text='50th Percentile', 
        annotation_position='bottom right',
        annotation=dict(font_size=20, font_family='Times New Roman', font_color='black'),
    )

    fig.add_vline(
        int(len(display_df.index) * 0.80),
        line_width=1,
        line_dash='dash',
        annotation_text='80th Percentile', 
        annotation_position='bottom right',
        annotation=dict(font_size=20, font_family='Times New Roman', font_color='black'),
    )

    return dcc.Graph(
        figure=fig
    )

# -------------------------------------------------------------------------------------------------------------

def certification_salaries_barchart():
    layout = go.Layout(
        margin=go.layout.Margin(
            l=150,   # left margin
            r=150,   # right margin
            b=100,   # bottom margin
            t=50    # top margin
        ), 
        height = 700,
        title_x = 0.5,
        title='Mean Incomes by Certification Type',
        xaxis_title="Credential",
        yaxis_title="Mean Income (CAD)",
        bargap=0
    )

    figure = go.Figure(
        layout=layout
    )

    dataframe = derived_df.loc[derived_df['Field of Study (CIP code)'].str.contains('00. Total')]
    dataframe = dataframe.loc[dataframe['Credential'] != 'Overall (All Graduates)']
    dataframe = dataframe.reset_index()

    print(dataframe)

    # The map of colors for each type of credential
    # NOTE: The credential types will be pulled in the order specified within the list.
    color_map = {
        'Certificate': 'red',
        'Diploma ': 'green',
        'Bachelor\'s degree': 'blue',
        'Professional bachelor\'s degree': 'yellow',
        'Bachelor\'s degree + certificate/diploma': 'grey',
        'Master\'s degree': 'orange',
        'Doctoral Degree': 'teal',
    }

    # Add each bar by certification type
    for certificate_name in color_map:
        # Get the new dataframe for this specific bar (by its credential name)
        bar_df = dataframe[dataframe["Credential"] == certificate_name]

        # Construct the new bar trace
        new_trace = go.Bar(
            x=bar_df["Credential"],
            y=bar_df['Average Income Ten Years After Graduation'],
            text = bar_df['Average Income Ten Years After Graduation'],
            textposition="inside",
            marker=dict(color='#024B7A'),
            marker_line=dict(width=1, color='black'),
            marker_color=color_map[certificate_name],
            width=0.5,
            hovertemplate='<extra></extra><br>Credential: %{x} <br>Average Median Income: %{y}',
            name=certificate_name
        )

        # Add the new bar trace into the overall figure
        figure.add_traces(new_trace)

    barChart = dcc.Graph(
        figure=figure
    )

    return barChart

# -------------------------------------------------------------------------------------------------------------

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
    html.Div(children=[
            "The ideal income, according to a ",
            html.A("study by Purdue University", href='https://www.purdue.edu/newsroom/releases/2018/Q1/money-only-buys-happiness-for-a-certain-amount.html'),
            " is $127K. They also note the emotional wellbeing threshold is 78K-$97K."
        ] 
    ),
    html.Div(id='Happiness-Scatterplot', children=jobs_happiness_scatterplot()),
    html.Div(id='Certification-Salaries-Barchart', children=certification_salaries_barchart())
])