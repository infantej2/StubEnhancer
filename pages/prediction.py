from .shared import generate_header

import dash
import pandas as pd
import numpy as np
import os
from dash import html, dcc, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import dash_cytoscape as cyto
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

"""
"""
def generate_nodes_ll(prefix_list, nodes):
  list_of_lists = []

  # Create amount of arrays matching prefix list
  for i in range (len(prefix_list)): list_of_lists.append([])

  for entry in nodes:
    entry_name = entry['data']['id']
    
    for i in range (len(prefix_list)):
      prefix = prefix_list[i]
      if entry_name.startswith(prefix):
        list_of_lists[i].append(entry_name)
        break

  return list_of_lists

# List of prefixes, each refrencing a layer of the neural network.
prefix_list = ['in', 'hl1n', 'hl2n', 'out']

nodes = [

    {
        'data': {'id':node_id, 'label':node_label},
        'position': {'x':x, 'y':y},
        'locked': True,
    }
    for node_id, node_label, x, y in (
        # Input nodes
        ("in1", "input-node1", 400, 100),
        ("in2", "input-node2", 400, 132.5),
        ("in3", "input-node3", 400, 165),
        # Hidden layer 1 nodes
        ("hl1n1", "hiddenlayer1-node1", 425, 75),
        ("hl1n2", "hiddenlayer1-node2", 425, 80),
        ("hl1n3", "hiddenlayer1-node3", 425, 85),
        ("hl1n4", "hiddenlayer1-node4", 425, 90),
        ("hl1n5", "hiddenlayer1-node5", 425, 95),
        ("hl1n6", "hiddenlayer1-node6", 425, 100),
        ("hl1n7", "hiddenlayer1-node7", 425, 105),
        ("hl1n8", "hiddenlayer1-node8", 425, 110),
        ("hl1n9", "hiddenlayer1-node9", 425, 115),
        ("hl1n10", "hiddenlayer1-node10", 425, 120),
        ("hl1n11", "hiddenlayer1-node11", 425, 125),
        ("hl1n12", "hiddenlayer1-node12", 425, 130),
        ("hl1n13", "hiddenlayer1-node13", 425, 135),
        ("hl1n14", "hiddenlayer1-node14", 425, 140),
        ("hl1n15", "hiddenlayer1-node15", 425, 145),
        ("hl1n16", "hiddenlayer1-node16", 425, 150),
        ("hl1n17", "hiddenlayer1-node17", 425, 155),
        ("hl1n18", "hiddenlayer1-node18", 425, 160),
        ("hl1n19", "hiddenlayer1-node19", 425, 165),
        ("hl1n20", "hiddenlayer1-node20", 425, 170),
        ("hl1n21", "hiddenlayer1-node21", 425, 175),
        ("hl1n22", "hiddenlayer1-node22", 425, 180),
        ("hl1n23", "hiddenlayer1-node23", 425, 185),
        ("hl1n24", "hiddenlayer1-node24", 425, 190),
        ("hl1n25", "hiddenlayer1-node25", 425, 195),
        # Hidden layer 2 nodes
        ("hl2n1", "hiddenlayer2-node1", 450, 115),
        ("hl2n2", "hiddenlayer2-node2", 450, 120),
        ("hl2n3", "hiddenlayer2-node3", 450, 125),
        ("hl2n4", "hiddenlayer2-node4", 450, 130),
        ("hl2n5", "hiddenlayer2-node5", 450, 135),
        ("hl2n6", "hiddenlayer2-node6", 450, 140),
        ("hl2n7", "hiddenlayer2-node7", 450, 145),
        ("hl2n8", "hiddenlayer2-node8", 450, 150),
        ("hl2n9", "hiddenlayer2-node9", 450, 155),
        ("hl2n10", "hiddenlayer2-node10", 450, 160),
        # Output neuron
        ("out1", "output-node1", 475, 132.5),
    )

]

nodes_lists = generate_nodes_ll(prefix_list, nodes)

def compute_node_edges(nodes_lists):
  edges = []

  for i in range(len(nodes_lists)):
    row = nodes_lists[i]

    # If there is no next row, ignore adding edges
    if i+1 >= len(nodes_lists):
      break

    for current in row:
      for next_row_entry in nodes_lists[i+1]:
        edges.append({'data': {'source':current, 'target':next_row_entry}})

  return edges


edges = compute_node_edges(nodes_lists)

# combine nodes and edges
elements = nodes+edges

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
        html.P(id="prediction_output", style={"color": "white", 'text-align':'left', 'width':'100%'})

    ]),
   
    # ==============================================================================
    html.Div(children=[
        cyto.Cytoscape(
            id="network-chart",
            layout={"name": "preset"},  # assign node positions ourselves
            style={"width":"100%", "height":"500px"},
            elements=elements,

            stylesheet=[    
            {
            'selector': 'node',
            'style': {
                'background-color': '#D84FD2',
                'width':"5%",
                'height':"5%"
                }
            },
            # style edges
            {
                'selector': 'edge',
                'style': {
                    'width':"0.5%"
                    }
            },

            ]
        ),


    ])

])


# ==============================================================================


@callback(
    Output(component_id='prediction_output', component_property='children'),
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
        
        return f"{prediction}"

    return None
         
  
    