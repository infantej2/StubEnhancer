"""
Program: prediction.py

Purpose: using a previously generated DNN for solving the linear regression at hand,
         create a network that represents its structure, allow users to select the 
         input vectors and display the results. Notably, this solution is hyper specific 
         to the data it is trained on and predictions have an RMSE of ~12,000. This is not an
         ideal number, but for learning purposes it will suffice.

Refrences:
    https://dash.plotly.com/cytoscape (for building the network graph)
"""
from .shared import generate_header, generate_navbar
import dash
import pandas as pd
import numpy as np
import os
from dash import html, dcc, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import dash_cytoscape as cyto
# this import often throws an error, so long as you have a compatible version of python however,
# it should work regardless.
from tensorflow.keras.models import load_model

dash.register_page(__name__)

"""
These are the encoded values (z-score) mapped to their actual value. The zscore is used
for normalizing the data.

Target encoding was used to convert these categorical values.

z-score represents how much you deviate from the mean in standard deviations.

"""
field_map = {
    'Agriculture, agriculture operations and related sciences': -0.594,
    'Natural resources and conservation': 0.171,
    'Architecture and related services': 0.436,
    'Area, ethnic, cultural, gender, and group studies': -0.917,
    'Communication, journalism and related programs': -0.315,
    'Communications technologies/technicians and support services': -1.021,
    'Computer and information sciences and support services': 0.576,
    'Personal and culinary services': -1.316,
    'Education': 1.871,
    'Engineering': 2.643,
    'Engineering technologies and engineering-related fields': 0.979,
    'Aboriginal and foreign languages, literatures and linguistics': -1.432,
    'Family and consumer sciences/human sciences': -1.144,
    'Legal professions and studies': 0.914,
    'English language and literature/letters': -0.611,
    'Liberal arts and sciences, general studies and humanities': -1.008,
    'Library science': -0.545,
    'Biological and biomedical sciences': -0.593,
    'Mathematics and statistics': 0.397,
    'Multidisciplinary/interdisciplinary studies': -0.0172,
    'Parks, recreation, leisure and fitness studies': -0.5426,
    'Philosophy and religious studies': -0.703,
    'Physical sciences': 0.465,
    'Science technologies/technicians': -0.407,
    'Psychology': 0.726,
    'Security and protective services': 0.974,
    'Public administration and social service professions': 0.740,
    'Social sciences': 0.164,
    'Construction trades': -0.839,
    'Mechanic and repair technologies/technicians': 2.238,
    'Precision production': 1.238,
    'Transportation and materials moving': -0.684,
    'Visual and performing arts': -1.475,
    'Health professions and related programs': 0.403,
    'Business, management, marketing and related support services': -0.0788,
    'History': -0.382,
    'French language and literature/lettersCAN': -0.307
}
# Years mapped to their respective encodings (zscore).
year_map = {
    1: -1.40,
    2: -0.70,
    3: 0.00,
    4: 0.71,
    5: 1.41
}
# Credentials mapped to an offset. This offset determines which column will be 
# turned on (set to 1). This is done this way because dummy-one-hot-encoding
# was used for these values.
cred_map = {
    "Certificate": 1,
    "Diploma": 2,
    "Bachelor's degree": 0,
    "Master's degree": 4,
    "Doctoral degree": 3,
    "Professional bachelor's degree": 5
}

dropdown_style = {"width": "100%", "align-items": "right", "margin-bottom":"10px"}
# the model was previously trained and here we load it.
Salary_model = load_model(os.path.join(".", "Salary_Model.h5"))

# ==============================================================================

df = pd.read_csv('./abSchool.csv')
# remove un-wanted characters from median income field
df['Median Income'] = df['Median Income'].str.replace('$', '')
df['Median Income'] = df['Median Income'].str.replace(',', '')
# convert median income string to a numeric value
df["Median Income"] = pd.to_numeric(df["Median Income"])

creds_list = list(df["Credential"].unique())
yrs_list = list(df["Years After Graduation"].unique())
df["Field of Study (2-digit CIP code)"] = df["Field of Study (2-digit CIP code)"].str.replace('[0-9]{2}. ', '', regex=True)
field_list = list(df["Field of Study (2-digit CIP code)"].unique())

"""
Function: generate_nodes_ll()

Purpose: this function is used to generate the nodes for the network. Networks can contain
         a plethora of nodes and thus some automation is necessary.

Parameters:
    prefix_list: a list of prefixes, these will allows us to assign
                 useful names to each neuron within the network.
    nodes: this is all the nodes that are defined according to cytoscape.
           nodes are represented as a list of dictionaries.
           Attributes of each node are also generally defined in dictionaries.

Return:
    list_of_lists: this a list of lists containing all the nodes with their 
                   assigned values.
"""
def generate_nodes_ll(prefix_list, nodes):
  list_of_lists = []

  # Create amount of arrays matching prefix list
  for i in range(len(prefix_list)):
    # create an empty lists of lists.
    list_of_lists.append([])

  # iterate through nodes.
  for entry in nodes:
    entry_name = entry['data']['id']

    for i in range(len(prefix_list)):
      prefix = prefix_list[i]
      if entry_name.startswith(prefix):
        list_of_lists[i].append(entry_name)
        break

  return list_of_lists


# List of prefixes, each refrencing a layer of the neural network.
prefix_list = ['in', 'hl1n', 'hl2n', 'out']
# define nodes.
nodes = [

    {
        'data': {'id': node_id, 'label': node_label},
        'position': {'x': x, 'y': y},
        'locked': True,
    }
    for node_id, node_label, x, y in (
        # Input nodes
        ("in1", "input-node1", 350, 100),
        ("in2", "input-node2", 350, 132.5),
        ("in3", "input-node3", 350, 165),
        # Hidden layer 1 nodes
        ("hl1n1", "hiddenlayer1-node1", 400, 75),
        ("hl1n2", "hiddenlayer1-node2", 400, 80),
        ("hl1n3", "hiddenlayer1-node3", 400, 85),
        ("hl1n4", "hiddenlayer1-node4", 400, 90),
        ("hl1n5", "hiddenlayer1-node5", 400, 95),
        ("hl1n6", "hiddenlayer1-node6", 400, 100),
        ("hl1n7", "hiddenlayer1-node7", 400, 105),
        ("hl1n8", "hiddenlayer1-node8", 400, 110),
        ("hl1n9", "hiddenlayer1-node9", 400, 115),
        ("hl1n10", "hiddenlayer1-node10", 400, 120),
        ("hl1n11", "hiddenlayer1-node11", 400, 125),
        ("hl1n12", "hiddenlayer1-node12", 400, 130),
        ("hl1n13", "hiddenlayer1-node13", 400, 135),
        ("hl1n14", "hiddenlayer1-node14", 400, 140),
        ("hl1n15", "hiddenlayer1-node15", 400, 145),
        ("hl1n16", "hiddenlayer1-node16", 400, 150),
        ("hl1n17", "hiddenlayer1-node17", 400, 155),
        ("hl1n18", "hiddenlayer1-node18", 400, 160),
        ("hl1n19", "hiddenlayer1-node19", 400, 165),
        ("hl1n20", "hiddenlayer1-node20", 400, 170),
        ("hl1n21", "hiddenlayer1-node21", 400, 175),
        ("hl1n22", "hiddenlayer1-node22", 400, 180),
        ("hl1n23", "hiddenlayer1-node23", 400, 185),
        ("hl1n24", "hiddenlayer1-node24", 400, 190),
        ("hl1n25", "hiddenlayer1-node25", 400, 195),
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
        ("out1", "output-node1", 500, 132.5),
    )

]

nodes_lists = generate_nodes_ll(prefix_list, nodes)

"""
Function: compute_node_edges()

Purpose: networks can contain many nodes, but they can contain even more edges,
         thus some automation for generating these edges is important.

Parameters:
    node_lists: takes a list of lists that contains node information.

Returns:
    edges: a list of edges, each edge has its properties defined in a dictionary, 
           it specifies a source and a target.
"""
def compute_node_edges(nodes_lists):
  # empty edge list.
  edges = []
  # the network being used is fully connected, and thus every 
  # neuron is connected to every other neuron through an edge in the following
  # layer.
  for i in range(len(nodes_lists)):
    row = nodes_lists[i]

    # If there is no next row, ignore adding edges
    if i+1 >= len(nodes_lists):
      break

    for current in row:
      for next_row_entry in nodes_lists[i+1]:
        edges.append({'data': {'source': current, 'target': next_row_entry}})

  return edges


edges = compute_node_edges(nodes_lists)

# combine nodes and edges to get the network graph.
elements = nodes+edges

node_weight_map = {

}

node_name_map = {
    'hl0n0': 'in1',
    #...
}

def update_element_values(elements, inputs):
    print(elements)

    pass

# LAYOUT
# ==============================================================================
layout = html.Div(className="body", children=[
    # generate_header(__name__),
    generate_navbar(__name__),
    html.H1(className="title", children='Prediction'),

    html.Div(children=[
        html.Div(style={'padding-top':'50px', 'padding-left':'250px', 'padding-right':'250px', 'display':'flex'}, children=[
            html.H5(className="prediction-three", id="prediction_output", style={
                       "color": "white", 'text-align': 'center', 'width': '100%'})
        ]),
        html.Div(className="prediction-wrapper", children=[
            html.Div(className="prediction-one", children=[
                html.H3("Enter details here:", style={'color':'white', 'text-align':'center'}),
                html.Br(),
                html.Div(children=[
                    html.H4("Credential Type", className="prediction-prompt"),
                    dcc.Dropdown(creds_list, id="input_creds", value="Select Credentials", multi=False, clearable=False,
                             style=dropdown_style),
                ], style={"display":"flex"}),
                html.Div(children=[
                    html.H4("Field of Study", className="prediction-prompt"),
                    dcc.Dropdown(field_list, id="input_field", value="Select Field", multi=False, clearable=False,
                     style=dropdown_style),
                ], style={"display":"flex"}),
                html.Div(children=[
                    html.H4("Years of Experience", className="prediction-prompt"),
                    dcc.Dropdown(yrs_list, id="input_years", value="Select Years Experience", multi=False, clearable=False,
                     style=dropdown_style),
                ], style={"display":"flex"}),
            ], style={"padding":"20px"}),
            html.Div(className="prediction-two", children=[
                html.Div(id='network-cytoscape')
            ])
        ])
    ]),
])

# CALLBACKS
# ==============================================================================
"""
Function: update_prediction_text()

Purpose: based on the callback input, run a novel prediction and update the output.

Parameters:
    credential_input: the choosen credentials. (string)
    field_input: the choosen field. (string)
    experience_input: the amount of experience in years, range [1-5]. (string)

Returns:
    None
"""
@callback(
    Output(component_id='prediction_output', component_property='children'),
    Input(component_id='input_creds', component_property='value'),
    Input(component_id='input_field', component_property='value'),
    Input(component_id='input_years', component_property='value')
)
def update_prediction_text(credential_input, field_input, experience_input) -> None:

    # check the credential input, map it to its encoding.
    if credential_input != "Select Credentials":
        credential_encoding = cred_map[credential_input]
    # check the field input, map it to its encoding.
    if field_input != "Select Field":
        field_encoding = field_map[field_input]
    # check the experience input, map it to its encoding.
    if experience_input != "Select Years Experience":
        year_encoding = year_map[experience_input]
    # once all have been selected, formulate the input vector.
    if (credential_input != "Select Credentials") and (field_input != "Select Field") and (experience_input != "Select Years Experience"):

        # sample vector is what will be passed to the neural network.
        sample_vector = [year_encoding, field_encoding, 0, 0, 0, 0, 0, 0]
        # credential encoding maps to an index. that index + 2 gives you the 0
        # to turn into a one.
        sample_vector[credential_encoding+2] = 1

        # EXAMPLE VECTOR: [yrs, field, bach, cert, dip, doc, mast, prof]
        sample_values = np.array([sample_vector], dtype=float)
        prediction = Salary_model.predict(sample_values)
        temp = float("{:.2f}".format(prediction[0][0]))
        formatted = "{:,}".format(temp)

        return html.H5(className="prediction-three", children=[f'According to your inputs, with a field of study in {field_input}, a credential type of {credential_input}, and {experience_input} years of experience, we predict that you can expect to earn ', html.Span(f'${formatted} CAD', style={'color':'#D84FD2'}), ' on average in Alberta.'])

    return "Enter details to get your prediction."

# ==============================================================================

@callback(
    Output(component_id='network-cytoscape', component_property='children'),
    Input(component_id='input_creds', component_property='value'),
    Input(component_id='input_field', component_property='value'),
    Input(component_id='input_years', component_property='value')
)
def update_network_cytoscape(credential_input, field_input, experience_input):
    credential_encoding = None
    field_encoding = None
    year_encoding = None

    # check the credential input, map it to its encoding.
    if credential_input != "Select Credentials":
        credential_encoding = cred_map[credential_input]
    # check the field input, map it to its encoding.
    if field_input != "Select Field":
        field_encoding = field_map[field_input]
    # check the experience input, map it to its encoding.
    if experience_input != "Select Years Experience":
        year_encoding = year_map[experience_input]

    if (credential_encoding != None) and (field_encoding != None) and (year_encoding != None):
        input_array = [year_encoding, field_encoding, 0, 0, 0, 0, 0, 0]
        input_array[credential_encoding+2] = 1

        global elements
        update_element_values(elements, input_array)

    return cyto.Cytoscape(
        id="network-chart",
        zoom=3,#zoom=2.5,
        # assign node positions ourselves
        layout={"name": "preset", "fit": False},
        style={"width": "100%", "height": "550px"},
        elements=elements,
        userZoomingEnabled=False,
        autoungrabify=True,
        autounselectify=True,
        pan={"x":-1000, "y":-180},# pan={"x":-850, "y":-130},
        panningEnabled=False,

        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'background-color': '#D84FD2',
                    'width': "5%",
                    'height': "5%"
                }
            },
            # style edges
            {
                'selector': 'edge',
                'style': {
                    'width': "0.1%" # 'width': "0.5%"
                }
            },
        ]
    )