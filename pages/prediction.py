from .shared import generate_header

import dash
import pandas as pd
import numpy as np
import os
from dash import html, dcc, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
from tensorflow.keras.models import load_model

dash.register_page(__name__)

# These are the encoded values, mapped to their actual values. 
field_map = {
'01. Agriculture, agriculture operations and related sciences':-0.594,
'03. Natural resources and conservation':0.171, 
'04. Architecture and related services':0.436, 
'05. Area, ethnic, cultural, gender, and group studies':-0.917,
'09. Communication, journalism and related programs':-0.315, 
'10. Communications technologies/technicians and support services':-1.021, 
'11. Computer and information sciences and support services':0.576, 
'12. Personal and culinary services':-1.316, 
'13. Education':1.871, 
'14. Engineering':2.643, 
'15. Engineering technologies and engineering-related fields':0.979, 
'16. Aboriginal and foreign languages, literatures and linguistics':-1.432, 
'19. Family and consumer sciences/human sciences':-1.144, 
'22. Legal professions and studies':0.914, 
'23. English language and literature/letters':-0.611, 
'24. Liberal arts and sciences, general studies and humanities':-1.008, 
'25. Library science':-0.545, 
'26. Biological and biomedical sciences':-0.593,
'27. Mathematics and statistics':0.397, 
'30. Multidisciplinary/interdisciplinary studies':-0.0172, 
'31. Parks, recreation, leisure and fitness studies':-0.5426, 
'38. Philosophy and religious studies':-0.703, 
'40. Physical sciences':0.465, 
'41. Science technologies/technicians':-0.407, 
'42. Psychology':0.726, 
'43. Security and protective services':0.974, 
'44. Public administration and social service professions':0.740, 
'45. Social sciences':0.164, 
'46. Construction trades':-0.839, 
'47. Mechanic and repair technologies/technicians':2.238, 
'48. Precision production':1.238, 
'49. Transportation and materials moving':-0.684, 
'50. Visual and performing arts':-1.475,
'51. Health professions and related programs':0.403, 
'52. Business, management, marketing and related support services':-0.0788,
'54. History':-0.382,
'55. French language and literature/lettersCAN':-0.307
}

# Years mapped to their respective encodings. 
year_map = {
1:-1.40,
2:-0.70,
3:0.00,
4:0.71,
5:1.41     
}
# Credentials mapped to an offet.
cred_map = {
"Certificate":1,
"Diploma":2,
"Bachelor's degree":0,
"Master's degree":4,
"Doctoral degree":3,
"Professional bachelor's degree":5
}


dropdown_style = {"width":"50%", "align-items":"center", 'margin-left':'30px'}
Salary_model = load_model(os.path.join(".","Salary_Model.h5"))

# ==============================================================================

df = pd.read_csv('./abSchool.csv')
# remove un-wanted characters from median income field
df['Median Income'] = df['Median Income'].str.replace('$','')
df['Median Income'] = df['Median Income'].str.replace(',','')
# convert median income string to a numeric value
df["Median Income"] = pd.to_numeric(df["Median Income"])



creds_list = list(df["Credential"].unique())
yrs_list = list(df["Years After Graduation"].unique())
field_list = list(df["Field of Study (2-digit CIP code)"].unique())

# ==============================================================================

layout = html.Div(className="body", children=[
    generate_header(__name__),
    html.H1(className="title", children='Stub Enhancer'),

    html.Div([

        dcc.Dropdown(creds_list, id="input_creds", value="Select Credentials", multi=False, clearable=False, 
        style=dropdown_style),
        dcc.Dropdown(field_list, id="input_field", value="Select Field", multi=False, clearable=False,
        style=dropdown_style),
        dcc.Dropdown(yrs_list, id="input_years", value="Select Years Experience", multi=False, clearable=False,
        style=dropdown_style),

    ]
    + [html.Div(id="dropdown-boxs",
                style={"color": "white", 'text-align':'left', 'width':'100%'})]
    , style={"marginTop":"50px"}),
    # ==============================================================================
     html.Div([
        html.H3(id="text-pred", style={"color": "white", "textAlign":"center", "paddingTop":"50px", "paddingBottom":"30px"}),

    ]
    + [html.Div(id="text-prediction",
                style={"color": "white", 'text-align':'left', 'width':'100%'})]
    ),

    # ==============================================================================
    """
    html.Div([
        dcc.Graph(id="scatter-plot"),

    ]
    + [html.Div(id="prediction-scatter-plot",
                style={"color": "white", 'text-align':'left', 'width':'100%'})]
    ),
    """
   # html.Footer(id="footer")
])


# ==============================================================================


@callback(
    Output(component_id='text-pred', component_property='children'),
    Input(component_id='input_creds', component_property='value'),
    Input(component_id='input_field', component_property='value'),
    Input(component_id='input_years', component_property='value')
)

def update_prediction_text(credential_input, field_input, experience_input):

    if credential_input != "Select Credentials":
        credential_encoding = cred_map[credential_input]
    
    if field_input != "Select Field": 
        field_encoding = field_map[field_input]

    if experience_input != "Select Years Experience":
        year_encoding = year_map[experience_input]
   
    if (credential_input != "Select Credentials") and (field_input != "Select Field") and (experience_input != "Select Years Experience"):

        # sample vector is what will be passed to the neural network.
        sample_vector = [year_encoding,field_encoding,0,0,0,0,0,0]
        # credential encoding maps to an index. that index + 2 gives you the 0 
        # to turn into a one.
        sample_vector[credential_encoding+2] = 1

        # EXAMPLE VECTOR: [yrs, field, bach, cert, dip, doc, mast, prof]
        sample_values = np.array( [sample_vector], dtype=float)
        prediction = Salary_model.predict(sample_values)
          # Graph layout
        layout = go.Layout(
            margin=go.layout.Margin(
                l=150,   # left margin
                r=150,   # right margin
                b=100,   # bottom margin
                t=50    # top margin
            ), 
            height = 700,
            title_x = 0.5,
            title='Expected Earning ',
            #xaxis_title=
            yaxis_title="Mean Income (CAD)",
            #bargap=0,
            #barmode='group',
            uniformtext_minsize=10,
            uniformtext_mode='show',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
        )

        fig = px.scatter(
        y=list(prediction),
        
        #y=list(pred),
        )

        fig.layout = layout
        
        return dcc.Graph(
            figure=fig
        )

"""
@callback(
    Output(component_id='prediction-scatter-plot', component_property='children'),
    Input(component_id='text-pred', component_property='value')
)
def prediction_scatter_plot(prediction):


    # Graph layout
    layout = go.Layout(
        margin=go.layout.Margin(
            l=150,   # left margin
            r=150,   # right margin
            b=100,   # bottom margin
            t=50    # top margin
        ), 
        height = 700,
        title_x = 0.5,
        title='Expected Earning ',
        #xaxis_title=
        yaxis_title="Mean Income (CAD)",
        #bargap=0,
        #barmode='group',
        uniformtext_minsize=10,
        uniformtext_mode='show',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
    )

    fig = px.scatter(
    y=list(prediction),
    
    #y=list(pred),
    )

    fig.layout = layout
    
    return dcc.Graph(
        figure=fig
    )
"""
  
    