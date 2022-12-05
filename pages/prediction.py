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
import copy

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

node_data = (
    # Input nodes
    ("hl0n0", "input-node1", 350, 165), # Years of Experience Input
    ("hl0n1", "input-node2", 350, 132.5), # Field of Study Input
    ("hl0n2", "input-node3", 350, 100), # Credential Type Input
    # Hidden layer 1 nodes
    ("hl1n0", "hiddenlayer1-node0", 400, 195),
    ("hl1n1", "hiddenlayer1-node1", 400, 190),
    ("hl1n2", "hiddenlayer1-node2", 400, 185),
    ("hl1n3", "hiddenlayer1-node3", 400, 180),
    ("hl1n4", "hiddenlayer1-node4", 400, 175),
    ("hl1n5", "hiddenlayer1-node5", 400, 170),
    ("hl1n6", "hiddenlayer1-node6", 400, 165),
    ("hl1n7", "hiddenlayer1-node7", 400, 160),
    ("hl1n8", "hiddenlayer1-node8", 400, 155),
    ("hl1n9", "hiddenlayer1-node9", 400, 150),
    ("hl1n10", "hiddenlayer1-node10", 400, 145),
    ("hl1n11", "hiddenlayer1-node11", 400, 140),
    ("hl1n12", "hiddenlayer1-node12", 400, 135),
    ("hl1n13", "hiddenlayer1-node13", 400, 130),
    ("hl1n14", "hiddenlayer1-node14", 400, 125),
    ("hl1n15", "hiddenlayer1-node15", 400, 120),
    ("hl1n16", "hiddenlayer1-node16", 400, 115),
    ("hl1n17", "hiddenlayer1-node17", 400, 110),
    ("hl1n18", "hiddenlayer1-node18", 400, 105),
    ("hl1n19", "hiddenlayer1-node19", 400, 100),
    ("hl1n20", "hiddenlayer1-node20", 400, 95),
    ("hl1n21", "hiddenlayer1-node21", 400, 90),
    ("hl1n22", "hiddenlayer1-node22", 400, 85),
    ("hl1n23", "hiddenlayer1-node23", 400, 80),
    ("hl1n24", "hiddenlayer1-node24", 400, 75),
    # Hidden layer 2 nodes
    ("hl2n0", "hiddenlayer2-node0", 450, 160),
    ("hl2n1", "hiddenlayer2-node1", 450, 155),
    ("hl2n2", "hiddenlayer2-node2", 450, 150),
    ("hl2n3", "hiddenlayer2-node3", 450, 145),
    ("hl2n4", "hiddenlayer2-node4", 450, 140),
    ("hl2n5", "hiddenlayer2-node5", 450, 135),
    ("hl2n6", "hiddenlayer2-node6", 450, 130),
    ("hl2n7", "hiddenlayer2-node7", 450, 125),
    ("hl2n8", "hiddenlayer2-node8", 450, 120),
    ("hl2n9", "hiddenlayer2-node9", 450, 115),
    # Output neuron
    ("hl3n0", "output-node1", 500, 132.5),
)

# List of prefixes, each refrencing a layer of the neural network.
prefix_list = ['hl0n', 'hl1n', 'hl2n', 'hl3n']
# define nodes.
nodes = [

    {
        'data': {'id': node_id, 'label': node_label},
        'position': {'x': x, 'y': y},
        'locked': True,
    }
    for node_id, node_label, x, y in node_data
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

node_bias_map = {
    'hl1n0': 6.30970573425293,
    'hl1n1': 6.373420715332031,
    'hl1n2': 6.2926411628723145,
    'hl1n3': 6.262844085693359,
    'hl1n4': 6.2598395347595215,
    'hl1n5': 6.366313457489014,
    'hl1n6': 6.351566791534424,
    'hl1n7': 6.226532936096191,
    'hl1n8': 6.292984962463379,
    'hl1n9': 6.2260212898254395,
    'hl1n10': 6.065385341644287,
    'hl1n11': 6.35082483291626,
    'hl1n12': 6.27856969833374,
    'hl1n13': 6.398648262023926,
    'hl1n14': 6.075347900390625,
    'hl1n15': -0.16542762517929077,
    'hl1n16': 6.307865142822266,
    'hl1n17': 6.227725028991699,
    'hl1n18': 6.3092041015625,
    'hl1n19': 6.220046043395996,
    'hl1n20': 6.250043869018555,
    'hl1n21': 6.35033655166626,
    'hl1n22': 6.133552074432373,
    'hl1n23': 6.372750759124756,
    'hl1n24': 6.261860370635986,

    'hl2n0': 5.208813190460205,
    'hl2n1': -0.1101112812757492,
    'hl2n2': 5.00095796585083,
    'hl2n3': 5.288653373718262,
    'hl2n4': 5.320930004119873,
    'hl2n5': -0.04685097560286522,
    'hl2n6': 4.839798450469971,
    'hl2n7': -0.051683176308870316,
    'hl2n8': -0.045348238199949265,
    'hl2n9': 0.0,

    'hl3n0': 3.5867397785186768
}

node_weight_map = {
    'hl0n0': {
        'hl1n0': 1.6998239755630493,
        'hl1n1': 1.6796207427978516,
        'hl1n2': -0.3652653694152832,
        'hl1n3': 2.092397928237915,
        'hl1n4': -0.10548891872167587,
        'hl1n5': 1.8947123289108276,
        'hl1n6': -0.3744759261608124,
        'hl1n7': 1.5063567161560059,
        'hl1n8': 1.7330899238586426,
        'hl1n9': 1.7733168601989746,
        'hl1n10': 1.5208772420883179,
        'hl1n11': 0.6301625967025757,
        'hl1n12': 1.0174931287765503,
        'hl1n13': 1.2394039630889893,
        'hl1n14': 1.4757050275802612,
        'hl1n15': -0.004975565243512392,
        'hl1n16': -0.3268454074859619,
        'hl1n17': 1.7774714231491089,
        'hl1n18': 1.0398503541946411,
        'hl1n19': 0.11324282735586166,
        'hl1n20': 0.29811763763427734,
        'hl1n21': 1.1621500253677368,
        'hl1n22': -0.3346680998802185,
        'hl1n23': 0.07421479374170303,
        'hl1n24': -0.8564237952232361
    },
    'hl0n1': {
        'hl1n0': 2.2272980213165283,
        'hl1n1': 2.855947732925415,
        'hl1n2': 2.8873729705810547,
        'hl1n3': 3.3228554725646973,
        'hl1n4': 2.7285213470458984,
        'hl1n5': 3.1511144638061523,
        'hl1n6': 2.4216620922088623,
        'hl1n7': 2.7983744144439697,
        'hl1n8': 3.321446180343628,
        'hl1n9': 2.6203742027282715,
        'hl1n10': 2.778169870376587,
        'hl1n11': 2.514295816421509,
        'hl1n12': 2.956851005554199,
        'hl1n13': 1.64446222782135,
        'hl1n14': 2.6757843494415283,
        'hl1n15': 0.004159396514296532,
        'hl1n16': 1.4940944910049438,
        'hl1n17': 3.045313835144043,
        'hl1n18': 3.15118670463562,
        'hl1n19': 2.607076406478882,
        'hl1n20': 3.017460584640503,
        'hl1n21': 2.866352081298828,
        'hl1n22': 0.38609376549720764,
        'hl1n23': 2.723710536956787,
        'hl1n24': 2.8148233890533447
    },
    'hl0n2': {
        'hl1n0': 5.008096694946289,
        'hl1n1': 4.768696308135986,
        'hl1n2': 5.258847713470459,
        'hl1n3': 5.444660186767578,
        'hl1n4': 4.835144519805908,
        'hl1n5': 5.031440734863281,
        'hl1n6': 5.133962631225586,
        'hl1n7': 5.324605464935303,
        'hl1n8': 4.946400165557861,
        'hl1n9': 4.974664688110352,
        'hl1n10': 5.2293314933776855,
        'hl1n11': 5.1116461753845215,
        'hl1n12': 5.389466285705566,
        'hl1n13': 4.853262901306152,
        'hl1n14': 5.119410991668701,
        'hl1n15': -0.3878220319747925,
        'hl1n16': 4.951801776885986,
        'hl1n17': 4.768928527832031,
        'hl1n18': 5.50830602645874,
        'hl1n19': 5.005943298339844,
        'hl1n20': 5.2899861335754395,
        'hl1n21': 5.291581153869629,
        'hl1n22': 4.620941638946533,
        'hl1n23': 5.181727409362793,
        'hl1n24': 4.91384220123291
    },
    'hl0n3': {
        'hl1n0': 3.0606915950775146,
        'hl1n1': 3.808816909790039,
        'hl1n2': 3.0748932361602783,
        'hl1n3': 3.0004966259002686,
        'hl1n4': 3.2281296253204346,
        'hl1n5': 3.554812431335449,
        'hl1n6': 3.4869749546051025,
        'hl1n7': 3.531054973602295,
        'hl1n8': 3.791715383529663,
        'hl1n9': 3.2920830249786377,
        'hl1n10': 3.345571994781494,
        'hl1n11': 3.3003926277160645,
        'hl1n12': 3.048734188079834,
        'hl1n13': 3.8009531497955322,
        'hl1n14': 3.303877830505371,
        'hl1n15': 0.13839289546012878,
        'hl1n16': 3.500737428665161,
        'hl1n17': 3.0144400596618652,
        'hl1n18': 3.086106777191162,
        'hl1n19': 3.5290510654449463,
        'hl1n20': 3.4185073375701904,
        'hl1n21': 3.6390838623046875,
        'hl1n22': 2.8924520015716553,
        'hl1n23': 3.2186787128448486,
        'hl1n24': 3.0064868927001953
    },
    'hl0n4': {
        'hl1n0': 4.63655424118042,
        'hl1n1': 4.689918041229248,
        'hl1n2': 4.326316833496094,
        'hl1n3': 4.841395378112793,
        'hl1n4': 4.809706687927246,
        'hl1n5': 4.816703796386719,
        'hl1n6': 4.596142292022705,
        'hl1n7': 4.855005741119385,
        'hl1n8': 4.431911468505859,
        'hl1n9': 4.566447734832764,
        'hl1n10': 4.643630504608154,
        'hl1n11': 4.895030975341797,
        'hl1n12': 4.4798784255981445,
        'hl1n13': 4.217167377471924,
        'hl1n14': 4.057007789611816,
        'hl1n15': -0.07734046876430511,
        'hl1n16': 4.7942728996276855,
        'hl1n17': 4.204549789428711,
        'hl1n18': 4.319786548614502,
        'hl1n19': 4.346503734588623,
        'hl1n20': 4.648456573486328,
        'hl1n21': 4.248529434204102,
        'hl1n22': 4.18317985534668,
        'hl1n23': 4.135735034942627,
        'hl1n24': 4.380845546722412
    },
    'hl0n5': {
        'hl1n0': 6.694617748260498,
        'hl1n1': 6.6269145011901855,
        'hl1n2': 6.43346643447876,
        'hl1n3': 6.608740329742432,
        'hl1n4': 7.112452030181885,
        'hl1n5': 6.807632923126221,
        'hl1n6': 6.758304595947266,
        'hl1n7': 6.379400730133057,
        'hl1n8': 6.963691234588623,
        'hl1n9': 6.7279229164123535,
        'hl1n10': 6.233376502990723,
        'hl1n11': 7.145421504974365,
        'hl1n12': 6.923112392425537,
        'hl1n13': 6.947080612182617,
        'hl1n14': 6.816265106201172,
        'hl1n15': -0.38863155245780945,
        'hl1n16': 7.146009922027588,
        'hl1n17': 6.83331823348999,
        'hl1n18': 7.141359806060791,
        'hl1n19': 6.459807872772217,
        'hl1n20': 6.352665901184082,
        'hl1n21': 6.560214042663574,
        'hl1n22': 6.327207565307617,
        'hl1n23': 6.807114601135254,
        'hl1n24': 6.409353256225586
    },
    'hl0n6': {
        'hl1n0': 7.832764148712158,
        'hl1n1': 7.5876383781433105,
        'hl1n2': 7.327826499938965,
        'hl1n3': 7.238029479980469,
        'hl1n4': 7.271841049194336,
        'hl1n5': 7.438954830169678,
        'hl1n6': 7.71069860458374,
        'hl1n7': 7.378063201904297,
        'hl1n8': 7.666310787200928,
        'hl1n9': 7.353804588317871,
        'hl1n10': 6.96175479888916,
        'hl1n11': 7.825803279876709,
        'hl1n12': 7.573156356811523,
        'hl1n13': 7.238348960876465,
        'hl1n14': 7.261570930480957,
        'hl1n15': -0.40716078877449036,
        'hl1n16': 7.2061357498168945,
        'hl1n17': 7.248673439025879,
        'hl1n18': 7.3093953132629395,
        'hl1n19': 7.757357120513916,
        'hl1n20': 7.221147537231445,
        'hl1n21': 7.5655317306518555,
        'hl1n22': 7.4408650398254395,
        'hl1n23': 7.735739231109619,
        'hl1n24': 7.189196586608887
    },
    'hl0n7': {
        'hl1n0': 5.0089545249938965,
        'hl1n1': 5.478814601898193,
        'hl1n2': 5.056637287139893,
        'hl1n3': 4.862729549407959,
        'hl1n4': 5.40625,
        'hl1n5': 4.792888164520264,
        'hl1n6': 5.224214553833008,
        'hl1n7': 5.24056339263916,
        'hl1n8': 5.09607458114624,
        'hl1n9': 4.944598197937012,
        'hl1n10': 5.186119556427002,
        'hl1n11': 5.511154651641846,
        'hl1n12': 5.42775297164917,
        'hl1n13': 5.277676582336426,
        'hl1n14': 5.159339427947998,
        'hl1n15': -0.1696135401725769,
        'hl1n16': 5.6219305992126465,
        'hl1n17': 5.309179306030273,
        'hl1n18': 4.9881486892700195,
        'hl1n19': 4.71476411819458,
        'hl1n20': 5.142502307891846,
        'hl1n21': 5.319896221160889,
        'hl1n22': 4.916578769683838,
        'hl1n23': 5.468764305114746,
        'hl1n24': 5.359665393829346
    },
    'hl1n0': {
        'hl2n0': 6.119061470031738,
        'hl2n1': 0.19321611523628235,
        'hl2n2': 6.2040252685546875,
        'hl2n3': 6.562792778015137,
        'hl2n4': 6.574466228485107,
        'hl2n5': -0.28601697087287903,
        'hl2n6': 6.071406364440918,
        'hl2n7': 0.04619581624865532,
        'hl2n8': -0.19809170067310333,
        'hl2n9': -0.06392869353294373
    },
    'hl1n1': {
        'hl2n0': 6.065148830413818,
        'hl2n1': 0.21840918064117432,
        'hl2n2': 5.755123615264893,
        'hl2n3': 6.187188148498535,
        'hl2n4': 6.491633415222168,
        'hl2n5': 0.34096622467041016,
        'hl2n6': 6.293137073516846,
        'hl2n7': 0.021169401705265045,
        'hl2n8': -0.3565886318683624,
        'hl2n9': 0.13821294903755188
    },
    'hl1n2': {
        'hl2n0': 6.579953193664551,
        'hl2n1': 0.21519234776496887,
        'hl2n2': 5.864772319793701,
        'hl2n3': 6.44788122177124,
        'hl2n4': 6.468542575836182,
        'hl2n5': 0.15765689313411713,
        'hl2n6': 6.378418922424316,
        'hl2n7': -0.05600070580840111,
        'hl2n8': 0.022318439558148384,
        'hl2n9': -0.053650349378585815
    },
    'hl1n3': {
        'hl2n0': 6.250358581542969,
        'hl2n1': -0.12248970568180084,
        'hl2n2': 5.830465316772461,
        'hl2n3': 6.408015251159668,
        'hl2n4': 6.533631324768066,
        'hl2n5': -0.42669594287872314,
        'hl2n6': 6.16141939163208,
        'hl2n7': 0.15053555369377136,
        'hl2n8': -0.016766492277383804,
        'hl2n9': 0.12287792563438416
    },
    'hl1n4': {
        'hl2n0': 6.320107936859131,
        'hl2n1': -0.12859371304512024,
        'hl2n2': 6.575395584106445,
        'hl2n3': 6.8184380531311035,
        'hl2n4': 6.053694725036621,
        'hl2n5': 0.18855169415473938,
        'hl2n6': 5.767554759979248,
        'hl2n7': -0.0064818598330020905,
        'hl2n8': 0.30228865146636963,
        'hl2n9': -0.31153255701065063
    },
    'hl1n5': {
        'hl2n0': 6.379042148590088,
        'hl2n1': 0.18436850607395172,
        'hl2n2': 5.889593124389648,
        'hl2n3': 6.011048793792725,
        'hl2n4': 6.269571781158447,
        'hl2n5': 0.17218424379825592,
        'hl2n6': 5.988518238067627,
        'hl2n7': -0.17780745029449463,
        'hl2n8': 0.29294905066490173,
        'hl2n9': -0.1101037859916687
    },
    'hl1n6': {
        'hl2n0': 5.969704627990723,
        'hl2n1': -0.309968501329422,
        'hl2n2': 6.428762435913086,
        'hl2n3': 6.455493927001953,
        'hl2n4': 6.625734806060791,
        'hl2n5': -0.05849173665046692,
        'hl2n6': 6.132472038269043,
        'hl2n7': -0.4278047978878021,
        'hl2n8': -0.18717685341835022,
        'hl2n9': -0.22527378797531128
    },
    'hl1n7': {
        'hl2n0': 6.470607757568359,
        'hl2n1': 0.12897710502147675,
        'hl2n2': 6.030381202697754,
        'hl2n3': 6.556962490081787,
        'hl2n4': 5.953242778778076,
        'hl2n5': -0.3719276189804077,
        'hl2n6': 5.82280158996582,
        'hl2n7': 0.3460972309112549,
        'hl2n8': 0.033487677574157715,
        'hl2n9': 0.25821754336357117
    },
    'hl1n8': {
        'hl2n0': 5.9221649169921875,
        'hl2n1': -1.823472666728776e-05,
        'hl2n2': 5.911160945892334,
        'hl2n3': 6.4218220710754395,
        'hl2n4': 6.353466510772705,
        'hl2n5': -0.24864985048770905,
        'hl2n6': 6.294277191162109,
        'hl2n7': -0.057940635830163956,
        'hl2n8': -0.04546266421675682,
        'hl2n9': -0.2777932584285736
    },
    'hl1n9': {
        'hl2n0': 6.342617511749268,
        'hl2n1': -0.10836600512266159,
        'hl2n2': 6.524314880371094,
        'hl2n3': 6.1320953369140625,
        'hl2n4': 6.428252696990967,
        'hl2n5': -0.1119590774178505,
        'hl2n6': 6.356146812438965,
        'hl2n7': -0.40898486971855164,
        'hl2n8': 0.27873679995536804,
        'hl2n9': -0.3350662291049957
    },
    'hl1n10': {
        'hl2n0': 5.901697635650635,
        'hl2n1': -0.23704740405082703,
        'hl2n2': 6.238559722900391,
        'hl2n3': 6.364165306091309,
        'hl2n4': 6.658803939819336,
        'hl2n5': -0.167593315243721,
        'hl2n6': 5.650622367858887,
        'hl2n7': -0.1921326220035553,
        'hl2n8': -0.13791145384311676,
        'hl2n9': -0.27701449394226074
    },
    'hl1n11': {
        'hl2n0': 6.045494079589844,
        'hl2n1': 0.06082528829574585,
        'hl2n2': 6.405332565307617,
        'hl2n3': 6.101964950561523,
        'hl2n4': 6.440366268157959,
        'hl2n5': -0.2981325387954712,
        'hl2n6': 5.998042583465576,
        'hl2n7': -0.3159220516681671,
        'hl2n8': -0.37233734130859375,
        'hl2n9': -0.3178344964981079
    },
    'hl1n12': {
        'hl2n0': 6.510704517364502,
        'hl2n1': -0.4677225947380066,
        'hl2n2': 6.319322109222412,
        'hl2n3': 6.395336151123047,
        'hl2n4': 6.32918643951416,
        'hl2n5': 0.15102733671665192,
        'hl2n6': 5.673569679260254,
        'hl2n7': -0.42916861176490784,
        'hl2n8': -0.4115724265575409,
        'hl2n9': -0.019120246171951294
    },
    'hl1n13': {
        'hl2n0': 6.087712287902832,
        'hl2n1': -0.18759223818778992,
        'hl2n2': 6.346996307373047,
        'hl2n3': 6.437861919403076,
        'hl2n4': 6.13913631439209,
        'hl2n5': -0.09562627971172333,
        'hl2n6': 6.477361679077148,
        'hl2n7': 0.04939303174614906,
        'hl2n8': -0.4120195209980011,
        'hl2n9': 0.11844328045845032
    },
    'hl1n14': {
        'hl2n0': 6.673402309417725,
        'hl2n1': -0.32453539967536926,
        'hl2n2': 6.029417037963867,
        'hl2n3': 6.359081745147705,
        'hl2n4': 6.655888080596924,
        'hl2n5': 0.03745441138744354,
        'hl2n6': 5.8005194664001465,
        'hl2n7': 0.26641708612442017,
        'hl2n8': -0.4281286895275116,
        'hl2n9': -0.12277805805206299
    },
    'hl1n15': {
        'hl2n0': -0.2588830888271332,
        'hl2n1': 0.1858678013086319,
        'hl2n2': -0.1622617244720459,
        'hl2n3': 0.0226537324488163,
        'hl2n4': 0.099312923848629,
        'hl2n5': -0.027857275679707527,
        'hl2n6': -0.035448867827653885,
        'hl2n7': -0.05882691964507103,
        'hl2n8': -0.19787496328353882,
        'hl2n9': -0.1091243326663971
    },
    'hl1n16': {
        'hl2n0': 6.266763687133789,
        'hl2n1': -0.3158339858055115,
        'hl2n2': 6.372068405151367,
        'hl2n3': 6.2682623863220215,
        'hl2n4': 6.5571675300598145,
        'hl2n5': 0.17552654445171356,
        'hl2n6': 6.209352016448975,
        'hl2n7': 0.06070869788527489,
        'hl2n8': -0.30090591311454773,
        'hl2n9': -0.387333482503891
    },
    'hl1n17': {
        'hl2n0': 6.650108337402344,
        'hl2n1': 0.1983458548784256,
        'hl2n2': 6.498725891113281,
        'hl2n3': 6.210233688354492,
        'hl2n4': 6.546604633331299,
        'hl2n5': -0.18111030757427216,
        'hl2n6': 6.4291205406188965,
        'hl2n7': 0.22734849154949188,
        'hl2n8': -0.29934224486351013,
        'hl2n9': 0.06106489896774292
    },
    'hl1n18': {
        'hl2n0': 6.0578203201293945,
        'hl2n1': -0.16220039129257202,
        'hl2n2': 5.718648433685303,
        'hl2n3': 6.421178817749023,
        'hl2n4': 6.532922744750977,
        'hl2n5': 0.21638910472393036,
        'hl2n6': 6.141354560852051,
        'hl2n7': 0.19830554723739624,
        'hl2n8': -0.0026128413155674934,
        'hl2n9': -0.0626453161239624
    },
    'hl1n19': {
        'hl2n0': 6.122926235198975,
        'hl2n1': -0.4464978575706482,
        'hl2n2': 6.636427402496338,
        'hl2n3': 6.309625148773193,
        'hl2n4': 6.755987167358398,
        'hl2n5': -0.3412724733352661,
        'hl2n6': 5.850965976715088,
        'hl2n7': 0.17335163056850433,
        'hl2n8': 0.3024897873401642,
        'hl2n9': 0.2384161651134491
    },
    'hl1n20': {
        'hl2n0': 6.294091701507568,
        'hl2n1': 0.17097914218902588,
        'hl2n2': 5.936381816864014,
        'hl2n3': 6.531039237976074,
        'hl2n4': 6.481112003326416,
        'hl2n5': 0.16221880912780762,
        'hl2n6': 5.937220573425293,
        'hl2n7': 0.006775972433388233,
        'hl2n8': -0.05589637905359268,
        'hl2n9': -0.26341724395751953
    },
    'hl1n21': {
        'hl2n0': 6.453196048736572,
        'hl2n1': -0.5531103610992432,
        'hl2n2': 5.774362564086914,
        'hl2n3': 5.962732315063477,
        'hl2n4': 5.935781955718994,
        'hl2n5': 0.0726763978600502,
        'hl2n6': 6.3755574226379395,
        'hl2n7': -0.13770882785320282,
        'hl2n8': 0.31332656741142273,
        'hl2n9': -0.23380589485168457
    },
    'hl1n22': {
        'hl2n0': 6.97260856628418,
        'hl2n1': -0.14250022172927856,
        'hl2n2': 6.346594333648682,
        'hl2n3': 6.572673320770264,
        'hl2n4': 6.61815881729126,
        'hl2n5': -0.15320438146591187,
        'hl2n6': 6.308144569396973,
        'hl2n7': 0.1160091757774353,
        'hl2n8': 0.2823553681373596,
        'hl2n9': 0.06806781888008118
    },
    'hl1n23': {
        'hl2n0': 6.493649482727051,
        'hl2n1': -0.12526020407676697,
        'hl2n2': 6.280045509338379,
        'hl2n3': 6.186079978942871,
        'hl2n4': 6.088979721069336,
        'hl2n5': -0.45654475688934326,
        'hl2n6': 6.098851680755615,
        'hl2n7': -0.21281564235687256,
        'hl2n8': -0.3657241463661194,
        'hl2n9': -0.18852341175079346
    },
    'hl1n24': {
        'hl2n0': 6.458211421966553,
        'hl2n1': -0.20836079120635986,
        'hl2n2': 6.455896854400635,
        'hl2n3': 6.483036994934082,
        'hl2n4': 6.451398849487305,
        'hl2n5': -0.06933707743883133,
        'hl2n6': 5.924823760986328,
        'hl2n7': 0.24349543452262878,
        'hl2n8': 0.3005892038345337,
        'hl2n9': 0.05491989850997925
    },
    'hl2n0': {
        'hl3n0': 6.5283284187316895
    },
    'hl2n1': {
        'hl3n0': -0.031764205545186996
    },
    'hl2n2': {
        'hl3n0': 6.8347954750061035
    },
    'hl2n3': {
        'hl3n0': 6.424211025238037
    },
    'hl2n4': {
        'hl3n0': 6.355916976928711
    },
    'hl2n5': {
        'hl3n0': -0.41954585909843445
    },
    'hl2n6': {
        'hl3n0': 7.096568584442139
    },
    'hl2n7': {
        'hl3n0': -0.42260774970054626
    },
    'hl2n8': {
        'hl3n0': -0.38760557770729065
    },
    'hl2n9': {
        'hl3n0': -0.47874927520751953
    }
}

'''
def generate_stylesheet(elements):
    stylesheet = [
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
    
    for element in elements:
        # Filter out edges

        # Insert new style for each node
        stylesheet.append()

    return stylesheet
'''

def update_element_values(elements, inputs, credential_encoding):
    # Generate name arrays, children of any given element will be all of the previous elements
    global nodes_lists
    real_nodes_list = copy.deepcopy(nodes_lists)
    # Update first list to include all input node names
    real_nodes_list[0] = ['hl0n0', 'hl0n1', 'hl0n2', 'hl0n3', 'hl0n4', 'hl0n5', 'hl0n6', 'hl0n7']
    
    values_map = {}

    # Compute first layer values with inputs
    for i in range(len(inputs)):
        values_map[real_nodes_list[0][i]] = inputs[i]

    # Make algorithm to automatically generate based on previous layer data
    for i in range(1, len(real_nodes_list)): # Start with 1 because we already dealt with the input layer
        for j in range(len(real_nodes_list[i])):
            # Get the current node we are looking at
            current_node = real_nodes_list[i][j]

            # Get all of the child nodes of the current node we are looking at (those of the previous layer)
            children_nodes = real_nodes_list[i - 1]

            # Calculate the weighted sums between each of the nodes (weight of current node given the node before it)
            weighted_sums = 0
            for child_node in children_nodes:
                weighted_sums += values_map[child_node] * node_weight_map[child_node][current_node]

            # Get the bias for the current node
            current_bias = node_bias_map[current_node]

            # The value given to the activation function is the sum of all of the weights*values of previous nodes + current bias
            pre_activation_function_value = weighted_sums + current_bias

            # RELU (rectified linear units) activation function, simply max(0, value)
            post_activation_function_value = np.maximum(0, pre_activation_function_value)

            # Set the value of the current node to the 
            values_map[current_node] = post_activation_function_value

    # Now that 'values_map' is populated with values for each and every node,
    # we will insert those values into the 'elements' displayed within the cytoscape
    for i in range(len(elements)):
        # Get the current element dictionary
        current_element = elements[i]

        # Ignore all edges; we only want nodes...
        if ('data' not in current_element) or ('id' not in current_element['data']): continue

        # We're going to do something special for the Credential Type input node ("hl0n2")
        # because it is actually supposed to represent 6 different input nodes (because of the encoding).
        # We will use the index value of the encoding itself for this node, to display in the Cytoscape,
        # then use the CSS styling to color it appropriately.
        if elements[i]['data']['id'] == 'hl0n2':
            # Set some data to the credential encoding index.
            # Values will be integers in range [1, 6]
            elements[i]['data']['cred_idx'] = (credential_encoding + 1)

        # Now that we know we know we have a node, simply grab the corresponding value for this node and insert it into the dict data
        # Also round it to two decimal points to make it a bit nicer
        elements[i]['data']['value'] = round(values_map[elements[i]['data']['id']], 2)

    return elements

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

        return html.H5(className="prediction-three", children=[f'According to your inputs, with a field of study in {field_input}, a credential type of {credential_input}, and {experience_input} years of experience, we predict that you can expect to earn ', html.Span(f'${formatted} CAD +/- $12,000 CAD', style={'color':'#D84FD2'}), ' on average in Alberta.'])

    return "Enter details to get your prediction."

# ==============================================================================

@callback(
    Output(component_id='network-cytoscape', component_property='children'),
    Input(component_id='input_creds', component_property='value'),
    Input(component_id='input_field', component_property='value'),
    Input(component_id='input_years', component_property='value')
)
def update_network_cytoscape(credential_input, field_input, experience_input):
    global elements

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

        elements = update_element_values(elements, input_array, credential_encoding)

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
                    #'label': 'data(value)',
                    'width': "5%",
                    'height': "5%"
                }
            },
            # STYLES FOR INDIVIDUAL NODE COLORS BELOW:

            {   # specific to 1 years exp
                'selector': '[value >= -1.4320]',
                'style': {
                    'background-color': '#7d47c2',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {   # specific to 2 years exp
                'selector': '[value >= -0.70]',
                'style': {
                    'background-color': '#6f3fac',
                    'width': "5%",
                    'height': "5%"
                }
            },
             {
                'selector': '[value >= 0]',
                'style': {
                    'background-color': '#613797',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {   # specific to 4 years exp
                'selector': '[value >= 0.71]',
                'style': {
                    'background-color': '#532f81',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {   # specific to 5 years exp
                'selector': '[value >= 1.41]',
                'style': {
                    'background-color': '#45276c',
                    'width': "5%",
                    'height': "5%"
                }
            },
           
            {
                'selector': '[value >= 5]',
                'style': {
                    'background-color': '#d0b8ef',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {
                'selector': '[value >= 7]',
                'style': {
                    'background-color': '#b995e7',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {
                'selector': '[value >= 10]',
                'style': {
                    'background-color': '#a272df',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {
                'selector': '[value >= 100]',
                'style': {
                    'background-color': '#8b4fd8',
                    'width': "5%",
                    'height': "5%"
                }
            },
            # STYLES FOR CREDENTIAL INPUT NODE
            {
                'selector': '[cred_idx=1]',
                'style': {
                    'background-color': '#7d47c2',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {
                'selector': '[cred_idx=2]',
                'style': {
                    'background-color': '#6f3fac',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {
                'selector': '[cred_idx=3]',
                'style': {
                    'background-color': '#613797',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {
                'selector': '[cred_idx=4]',
                'style': {
                    'background-color': '#532f81',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {
                'selector': '[cred_idx=5]',
                'style': {
                    'background-color': '#45276c',
                    'width': "5%",
                    'height': "5%"
                }
            },
            {
                'selector': '[cred_idx=6]',
                'style': {
                    'background-color': '#371f56',
                    'width': "5%",
                    'height': "5%"
                }
            },
            # STYLE FOR OUTPUT NODES
            {
                'selector': '[id^="hl3"]', # Specific to hidden layer 3 nodes; aka output nodes
                'style': {
                    'label': 'data(value)',
                    'font-size': '0.3em',
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

'''
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
'''