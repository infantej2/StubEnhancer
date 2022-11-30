from .shared import generate_header, generate_navbar

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html, callback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

dash.register_page(__name__)

# -------------------------------------------------------------------------------------------------------------

# Dataframe cleaning and prep
'''
df = pd.read_csv('./abSchool.csv')
df = df.dropna()
df = df.drop(columns=['Cohort Size'])
df['Median Income'] = df['Median Income'].str.replace('[$,]', '', regex=True).astype(int)
dflist = df.loc[:,"Field of Study (2-digit CIP code)"]
list = dflist.to_numpy()
list = np.unique(list)
'''

derived_df = pd.read_csv('./derived_data.csv')
dflist = derived_df[derived_df['Field of Study (CIP code)'].str.contains('[0-9]{2}.[0-9]{2}', regex=True) == False]
dflist = dflist[derived_df['Field of Study (CIP code)'] != '00. Total']
dflist = dflist.loc[:,'Field of Study (CIP code)']
list = np.unique(dflist.to_numpy())

# -------------------------------------------------------------------------------------------------------------


layout = html.Div(className="body", children=[
    generate_navbar(__name__),
    #generate_header(__name__),
    html.H1(className="title", children='By Field', style={'text-align':'center'}),
    html.Div(children=[
        html.Div(children=[
            html.Div(
                html.H4("Field of Study",
                style={"color": "white", 'text-align':'left', 'width':'100%'}),
            ),
            dcc.Dropdown(list, id="FoS", value="11. Computer and information sciences and support services", multi=False,
            style={'width':'90%', 'margin-left':'30px'})
        ], style={"display":"flex", "width":"70%", "margin":"auto", "paddingTop":"40px"}),
        html.H5(id='FoS-Salary-Text',
        style={"color": "white", "textAlign":"center", "paddingTop":"50px", "paddingBottom":"30px"}),
        html.Div(className="field-wrapper", children=[
            html.Div(className="field-one", children=[
                html.Div(id='FoS-Yearly-Salary-Linechart')
        ], style={"paddingTop":"20px"}),
            html.Div(className="field-two", children=[
                html.Div(id='FoS-Certification-Graph')
        ], style={"paddingTop":"20px"}),
        ])
    ])
])

# --------------------------------------------------------------------------------------------------------------------------------------

years_texts = [
    'Ten Years After Graduation',
    'Five Years After Graduation',
    'Two Years After Graduation',
    'One Year After Graduation'
]

years_numbers = [
    10,
    5,
    2,
    1
]

def df_has_year_salary(dataframe, years):
    has_year_salary = False

    for index, row in dataframe.iterrows():
        salary = row[f'Average Income {years}']
        size = row[f'Cohort Size {years}']

        # Ignore N/A rows
        if pd.isna(salary) or pd.isna(size):
            continue

        has_year_salary = ((salary != 0) and (size != 0))
        if has_year_salary:
            break

    return has_year_salary

def df_avg_salary_smooth_sum(dataframe, years):
    total_salary_sum = 0
    total_people = 0

    for index, row in dataframe.iterrows():
        salary = row[f'Average Income {years}']
        size = row[f'Cohort Size {years}']

        # Ignore N/A rows
        if pd.isna(salary) or pd.isna(size):
            continue

        total_salary_sum += (salary * size)
        total_people += size

    expected_salary = 0

    # Prevent divide by 0 when no data
    if total_people > 0:
       expected_salary = int(total_salary_sum / total_people)

    return expected_salary

# --------------------------------------------------------------------------------------------------------------------------------------

@callback(
    Output(component_id='FoS-Salary-Text', component_property='children'),
    Input(component_id='FoS', component_property='value')
)
def update_fos_salary_text(field_of_study):
    if not field_of_study:
        return 'Select a Field of Study for an Average Salary...'

    fos_split = field_of_study.split('. ')
    fos_code = fos_split[0]
    fos_name = fos_split[-1]

    dataframe = derived_df.loc[derived_df['Field of Study (CIP code)'].str.startswith(fos_code)]
    dataframe = dataframe.loc[dataframe['Credential'] != 'Overall (All Graduates)']
    #dataframe = dataframe.dropna(axis=0)

    expected_salary = 0
    salary_year = years_texts[-1]

    # Loop each year (descending), when we have a salary, exit
    for year in years_texts:
        salary_year = year
        expected_salary = df_avg_salary_smooth_sum(dataframe, salary_year)

        # As soon as we have a salary, exit
        if expected_salary != 0:
            break

    return f'On average, those studying \"{fos_name}\" can expect to make ~${expected_salary} CAD {salary_year.lower()}.'

# --------------------------------------------------------------------------------------------------------------------------------------

@callback(
    Output(component_id='FoS-Yearly-Salary-Linechart', component_property='children'),
    Input(component_id='FoS', component_property='value')
)
def update_fos_salary_linechart(field_of_study):
    layout = go.Layout(
        margin=go.layout.Margin(
            l=150,   # left margin
            r=150,   # right margin
            b=100,   # bottom margin
            t=50    # top margin
        ), 
        height = 700,
        title_x = 0.5,
        title='Average Earnings Over Time By Years After Graduation per Job Title Within the Chosen Field',
        xaxis_title='Years After Graduation',
        yaxis_title='Average Salary (CAD)',
        bargap=0,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
    )

    if not field_of_study:
        return dcc.Graph(
            figure=go.Figure(
                layout=layout
            )
        )

    if field_of_study == '00. Total (All Graduates)':
        return

    fos_split = field_of_study.split('. ')
    fos_code = fos_split[0]
    fos_name = fos_split[-1]

    dataframe = derived_df.loc[derived_df['Credential'] == 'Overall (All Graduates)']
    dataframe = dataframe.loc[dataframe['Field of Study (CIP code)'].str.startswith(fos_code)]

    new_df_list = []
    for index, row in dataframe.iterrows():
        for i in range(len(years_texts)):
            year_text = years_texts[i]
            year_number = years_numbers[i]

            fos = row['Field of Study (CIP code)']
            column_name = f'Average Income {year_text}'
            year_salary = row[column_name]

            if pd.isna(year_salary): continue

            new_df_list.append({
                'field_of_study': fos,
                'year': year_number,
                'salary': year_salary
            })

    new_df = pd.DataFrame(new_df_list)

    figure = px.line(
        data_frame=new_df,
        x=new_df['year'],
        y=new_df['salary'],
        color='field_of_study',
        markers=True
    )

    figure.layout = layout
    
    figure.update_traces(
        hovertemplate = 'Years After Graduation: %{x}<br>Average Salary (CAD): %{y}',
    )

    return dcc.Graph(
        figure=figure
    )

# --------------------------------------------------------------------------------------------------------------------------------------

# Explore 2-digit CIP update and callback
@callback(
    Output(component_id='FoS-Certification-Graph', component_property='children'),
    Input(component_id='FoS', component_property='value')
)
def update_fos_certification_graph(field_of_study):
    layout = go.Layout(
        margin=go.layout.Margin(
            l=150,   # left margin
            r=150,   # right margin
            b=100,   # bottom margin
            t=50    # top margin
        ), 
        height = 700,
        title_x = 0.5,
        title='Mean Incomes by Certification Type for the Chosen Field',
        xaxis_title="Credential",
        yaxis_title="Mean Income (CAD)",
        bargap=0,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
    )

    figure = go.Figure(
        layout=layout
    )

    if not field_of_study:
        return dcc.Graph(figure=figure)

    fos_split = field_of_study.split('. ')
    fos_code = fos_split[0]

    '''
    temp_df1 = (df.loc[df['Field of Study (4-digit CIP code)'].str.lower().str.contains(field_of_study[0-2])])
    df1 = temp_df1.groupby(['Field of Study (4-digit CIP code)']).mean(numeric_only=True).astype(int).sort_values('Median Income', ascending=True)
    df1.reset_index(inplace=True)
    '''

    '''
    temp_df2 = (df.loc[df['Field of Study (2-digit CIP code)'] == field_of_study])
    df2 = temp_df2.groupby(['Credential']).mean(numeric_only=True).astype(int).sort_values('Median Income', ascending=True)
    df2.reset_index(inplace=True)
    '''

    temp_df2 = derived_df.loc[derived_df['Field of Study (CIP code)'].str.startswith(fos_code)]
    temp_df2 = temp_df2.loc[temp_df2['Credential'] != 'Overall (All Graduates)']
    df2 = temp_df2.reset_index()

    income_year_column = 'Average Income Ten Years After Graduation'
    for years in years_texts:
        if df_has_year_salary(df2, years):
            income_year_column = f'Average Income {years}'
            break

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
        bar_df = df2[df2["Credential"] == certificate_name]

        # Ignore this bar if there are no entries in the new dataframe
        if bar_df.empty: continue
        if bar_df[income_year_column].isnull().values.any(): continue

        # Construct the new bar trace
        new_trace = go.Bar(
            x=bar_df["Credential"],
            y=bar_df[income_year_column], # y=bar_df["Median Income"],
            text = bar_df[income_year_column],
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

# --------------------------------------------------------------------------------------------------------------------------------------