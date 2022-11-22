from .shared import generate_header

import dash
import pandas as pd
import numpy as np
import os
from dash import html, dcc, Input, Output, callback
from tensorflow.keras.models import load_model

dash.register_page(__name__)

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

year_map = {
1:-1.40,
2:-0.70,
3:0.00,
4:0.71,
5:1.41    
}

cred_map = {
"Certificate":1,
"Diploma":2,
"Bachelor's Degree":0,
"Master's Degree":4,
"Doctoral Degree":3,
"Professional bachelors degree":5
}


dropdown_style = {"width":"50%", "align-items":"center", 'margin-left':'30px'}
Salary_model = load_model(os.path.join(".","Salary_Model.h5"))

# ==============================================================================

df = pd.read_csv('./abSchool.csv')
df = df.dropna()
creds_list = list(df["Credential"].unique())
yrs_list = list(df["Years After Graduation"].unique())
field_list = list(df["Field of Study (2-digit CIP code)"].unique())

# ==============================================================================

layout = html.Div(className="body", children=[
    generate_header(__name__),
    html.H1(className="title", children='Stub Enhancer'),

    html.Div([

        dcc.Dropdown(creds_list, id="input_creds", value="Select Credentials", multi=False, 
        style=dropdown_style),
        dcc.Dropdown(field_list, id="input_field", value="Select Field", multi=False,
        style=dropdown_style),
        dcc.Dropdown(yrs_list, id="input_years", value="Select Years Experience", multi=False,
        style=dropdown_style)

    ]
    + [html.Div(id="output_pred",
                style={"color": "white", 'text-align':'left', 'width':'100%'})]
    , style={"marginTop":"50px"}),

    html.Div([

    ]
    + [html.Div(id="output_pred1",
                style={"color": "white", 'text-align':'left', 'width':'100%'})]
    ),
    html.Footer(id="footer")
])


# ==============================================================================

@callback(
    Output(component_id='output_pred1', component_property='children'),
    Input(component_id='input_creds', component_property='value'),
    Input(component_id='input_field', component_property='value'),
    Input(component_id='input_years', component_property='value')
)

def update_output(input1, input2, input3):


    if input1 != "Select Credentials":
        credential_encoding = cred_map[input1]
    
    if input2 != "Select Field": 
        field_encoding = field_map[input2]

    if input3 != "Select Years Experience":
        year_encoding = year_map[input3]
   
    if (input1 != "Select Credentials") and (input2 != "Select Field") and (input3 != "Select Years Experience"):
        sample_vector = [year_encoding,field_encoding,0,0,0,0,0,0]

        sample_vector[credential_encoding+2] = 1

        #[yrs, field, bach, cert, dip, doc, mast, prof]

   
        sample_values = np.array( [sample_vector],\
            dtype=float)
        pred = Salary_model.predict(sample_values)
        #print(pred)
        #pred = np.argmax(pred,axis=1)

        return f'Input 1 {credential_encoding}, Input 2 {field_encoding}, and Input 3 {year_encoding} give {pred} +/- $12,000'
    