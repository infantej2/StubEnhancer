from .shared import generate_header

import dash
import pandas as pd
import os
from dash import html, dcc, Input, Output, callback
from tensorflow.keras.models import load_model

dash.register_page(__name__)

dropdown_style = {"width":"50%", "disaply":"flex", "align-items":"center", 'margin-left':'30px', 'margin-top':'100px'}
Salary_model = load_model(os.path.join(".","Salary_Model.h5"))
# ==============================================================================

df = pd.read_csv('./abSchool.csv')
df = df.dropna()
creds_list = list(df["Credential"].unique())
yrs_list = list(df["Years After Graduation"].unique())
field_list = list(df["Field of Study (2-digit CIP code)"].unique())

# ==============================================================================

layout = html.Div(children=[
    generate_header(__name__),
    html.H1(children='Prediction'),

    html.Div([

        dcc.Dropdown(creds_list, id="input_creds", value="Select Credentials", multi=False, 
        style=dropdown_style),
        dcc.Dropdown(field_list, id="input_field", value="Select Field", multi=False,
        style=dropdown_style),
        dcc.Dropdown(yrs_list, id="input_years", value="Select Years Experience", multi=False,
        style=dropdown_style)

    ]
    + [html.Div(id="output_pred")]
    ),

    html.Div([

    ]
    + [html.Div(id="output_pred1")]
    )

])


# ==============================================================================

@callback(
    Output(component_id='output_pred1', component_property='children'),
    Input(component_id='input_creds', component_property='value'),
    Input(component_id='input_field', component_property='value'),
    Input(component_id='input_years', component_property='value')
)

def update_output(input1, input2, input3):

    # map input 1 to encoding
    # map input 2 to encoding
    # map input 3 to encoding
    return u'Input 1 {}, Input 2 {}, and Input 3 {}'.format(input1, input2, input3)
    