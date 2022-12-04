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
prefix_list = ['hl0n', 'hl1n', 'hl2n', 'hl3n']
# define nodes.
nodes = [

    {
        'data': {'id': node_id, 'label': node_label},
        'position': {'x': x, 'y': y},
        'locked': True,
    }
    for node_id, node_label, x, y in (
        # Input nodes
        ("hl0n0", "input-node1", 350, 100),
        ("hl0n1", "input-node2", 350, 132.5),
        ("hl0n2", "input-node3", 350, 165),
        # Hidden layer 1 nodes
        ("hl1n0", "hiddenlayer1-node0", 400, 75),
        ("hl1n1", "hiddenlayer1-node1", 400, 80),
        ("hl1n2", "hiddenlayer1-node2", 400, 85),
        ("hl1n3", "hiddenlayer1-node3", 400, 90),
        ("hl1n4", "hiddenlayer1-node4", 400, 95),
        ("hl1n5", "hiddenlayer1-node5", 400, 100),
        ("hl1n6", "hiddenlayer1-node6", 400, 105),
        ("hl1n7", "hiddenlayer1-node7", 400, 110),
        ("hl1n8", "hiddenlayer1-node8", 400, 115),
        ("hl1n9", "hiddenlayer1-node9", 400, 120),
        ("hl1n10", "hiddenlayer1-node10", 400, 125),
        ("hl1n11", "hiddenlayer1-node11", 400, 130),
        ("hl1n12", "hiddenlayer1-node12", 400, 135),
        ("hl1n13", "hiddenlayer1-node13", 400, 140),
        ("hl1n14", "hiddenlayer1-node14", 400, 145),
        ("hl1n15", "hiddenlayer1-node15", 400, 150),
        ("hl1n16", "hiddenlayer1-node16", 400, 155),
        ("hl1n17", "hiddenlayer1-node17", 400, 160),
        ("hl1n18", "hiddenlayer1-node18", 400, 165),
        ("hl1n19", "hiddenlayer1-node19", 400, 170),
        ("hl1n20", "hiddenlayer1-node20", 400, 175),
        ("hl1n21", "hiddenlayer1-node21", 400, 180),
        ("hl1n22", "hiddenlayer1-node22", 400, 185),
        ("hl1n23", "hiddenlayer1-node23", 400, 190),
        ("hl1n24", "hiddenlayer1-node24", 400, 195),
        # Hidden layer 2 nodes
        ("hl2n0", "hiddenlayer2-node0", 450, 115),
        ("hl2n1", "hiddenlayer2-node1", 450, 120),
        ("hl2n2", "hiddenlayer2-node2", 450, 125),
        ("hl2n3", "hiddenlayer2-node3", 450, 130),
        ("hl2n4", "hiddenlayer2-node4", 450, 135),
        ("hl2n5", "hiddenlayer2-node5", 450, 140),
        ("hl2n6", "hiddenlayer2-node6", 450, 145),
        ("hl2n7", "hiddenlayer2-node7", 450, 150),
        ("hl2n8", "hiddenlayer2-node8", 450, 155),
        ("hl2n9", "hiddenlayer2-node9", 450, 160),
        # Output neuron
        ("hl3n0", "output-node1", 500, 132.5),
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

node_bias_map = {
    'hl1n0': 5.829582214355469,
    'hl1n1': 5.752955913543701,
    'hl1n2': 5.9115753173828125,
    'hl1n3': 5.739481449127197,
    'hl1n4': 5.940816879272461,
    'hl1n5': 5.843386173248291,
    'hl1n6': 5.939788341522217,
    'hl1n7': 5.896178722381592,
    'hl1n8': 5.790760517120361,
    'hl1n9': 5.881162166595459,
    'hl1n10': 5.879663467407227,
    'hl1n11': 5.8310699462890625,
    'hl1n12': 5.703884601593018,
    'hl1n13': 5.881357669830322,
    'hl1n14': 5.8995137214660645,
    'hl1n15': 5.945316314697266,
    'hl1n16': 5.4350690841674805,
    'hl1n17': 5.785948753356934,
    'hl1n18': 5.920124530792236,
    'hl1n19': 5.753687858581543,
    'hl1n20': 5.828001499176025,
    'hl1n21': 5.735006332397461,
    'hl1n22': 5.790147304534912,
    'hl1n23': 5.862607955932617,
    'hl1n24': 5.82393217086792,

    'hl2n0': 4.824225425720215,
    'hl2n1': 4.690515995025635,
    'hl2n2': -0.06880684942007065,
    'hl2n3': -0.038480594754219055,
    'hl2n4': 4.789127826690674,
    'hl2n5': -0.026411497965455055,
    'hl2n6': 4.556331634521484,
    'hl2n7': -0.05570553243160248,
    'hl2n8': 4.469769477844238,
    'hl2n9': 4.802062511444092,

    'hl3n0': 3.2540688514709473
}

node_weight_map = {
    'hl0n0': {
        'hl1n0': '1.1306060552597046',
        'hl1n1': '-0.5818613767623901',
        'hl1n2': '1.2690565586090088',
        'hl1n3': '1.04196298122406',
        'hl1n4': '1.4295700788497925',
        'hl1n5': '-0.9071117043495178',
        'hl1n6': '-0.7772510051727295',
        'hl1n7': '0.17188766598701477',
        'hl1n8': '1.8364285230636597',
        'hl1n9': '-0.5317198038101196',
        'hl1n10': '1.4037822484970093',
        'hl1n11': '0.7294248938560486',
        'hl1n12': '-0.14708212018013',
        'hl1n13': '1.2385333776474',
        'hl1n14': '1.039716124534607',
        'hl1n15': '1.8448635339736938',
        'hl1n16': '-0.16965647041797638',
        'hl1n17': '1.2111872434616089',
        'hl1n18': '1.6737779378890991',
        'hl1n19': '-0.16768713295459747',
        'hl1n20': '2.051917791366577',
        'hl1n21': '0.17914795875549316',
        'hl1n22': '0.8457170128822327',
        'hl1n23': '1.608885645866394',
        'hl1n24': '0.6010625958442688'
    },
    'hl0n1': {
        'hl1n0': '2.052727699279785',
        'hl1n1': '2.439629316329956',
        'hl1n2': '1.1631901264190674',
        'hl1n3': '2.583994150161743',
        'hl1n4': '1.0012441873550415',
        'hl1n5': '2.9418959617614746',
        'hl1n6': '1.7836843729019165',
        'hl1n7': '2.846224308013916',
        'hl1n8': '2.907182455062866',
        'hl1n9': '2.941953420639038',
        'hl1n10': '2.479060649871826',
        'hl1n11': '2.4329118728637695',
        'hl1n12': '1.0719285011291504',
        'hl1n13': '1.0144392251968384',
        'hl1n14': '1.732251524925232',
        'hl1n15': '3.071824550628662',
        'hl1n16': '2.45906925201416',
        'hl1n17': '2.7050435543060303',
        'hl1n18': '2.542598247528076',
        'hl1n19': '2.4875073432922363',
        'hl1n20': '3.1038901805877686',
        'hl1n21': '2.454641580581665',
        'hl1n22': '2.5528502464294434',
        'hl1n23': '2.9761626720428467',
        'hl1n24': '2.6325700283050537'
    },
    'hl0n2': {
        'hl1n0': '4.669046878814697',
        'hl1n1': '4.836755752563477',
        'hl1n2': '5.101151466369629',
        'hl1n3': '4.973818302154541',
        'hl1n4': '4.57702112197876',
        'hl1n5': '4.522152423858643',
        'hl1n6': '4.432144641876221',
        'hl1n7': '4.7111663818359375',
        'hl1n8': '4.368005752563477',
        'hl1n9': '4.818190574645996',
        'hl1n10': '5.129016876220703',
        'hl1n11': '5.047948360443115',
        'hl1n12': '4.788633823394775',
        'hl1n13': '4.6514058113098145',
        'hl1n14': '4.618426322937012',
        'hl1n15': '4.716238021850586',
        'hl1n16': '4.4866228103637695',
        'hl1n17': '4.741394996643066',
        'hl1n18': '4.466028213500977',
        'hl1n19': '4.689206600189209',
        'hl1n20': '4.901899814605713',
        'hl1n21': '4.699787616729736',
        'hl1n22': '4.953372478485107',
        'hl1n23': '4.640087604522705',
        'hl1n24': '4.544368267059326'
    },
    'hl0n3': {
        'hl1n0': '2.802384614944458',
        'hl1n1': '2.969755172729492',
        'hl1n2': '3.3608152866363525',
        'hl1n3': '3.139008045196533',
        'hl1n4': '2.9994118213653564',
        'hl1n5': '3.221187114715576',
        'hl1n6': '2.912064790725708',
        'hl1n7': '2.704968214035034',
        'hl1n8': '3.1727821826934814',
        'hl1n9': '3.0938096046447754',
        'hl1n10': '3.244013547897339',
        'hl1n11': '3.4364891052246094',
        'hl1n12': '3.246084213256836',
        'hl1n13': '2.97571063041687',
        'hl1n14': '2.869110345840454',
        'hl1n15': '2.8350155353546143',
        'hl1n16': '2.5995707511901855',
        'hl1n17': '3.3723795413970947',
        'hl1n18': '3.3230438232421875',
        'hl1n19': '2.746809482574463',
        'hl1n20': '3.0568649768829346',
        'hl1n21': '3.48445987701416',
        'hl1n22': '2.9418911933898926',
        'hl1n23': '3.333613157272339',
        'hl1n24': '2.848719358444214'
    },
    'hl0n4': {
        'hl1n0': '3.8329391479492188',
        'hl1n1': '3.99564528465271',
        'hl1n2': '4.484632968902588',
        'hl1n3': '4.174055576324463',
        'hl1n4': '3.847280263900757',
        'hl1n5': '4.456079959869385',
        'hl1n6': '3.7610456943511963',
        'hl1n7': '3.8027381896972656',
        'hl1n8': '4.082535743713379',
        'hl1n9': '4.219224452972412',
        'hl1n10': '3.8628857135772705',
        'hl1n11': '3.702907085418701',
        'hl1n12': '4.182790279388428',
        'hl1n13': '3.7413747310638428',
        'hl1n14': '4.299930572509766',
        'hl1n15': '4.352145195007324',
        'hl1n16': '3.6906204223632812',
        'hl1n17': '4.2786545753479',
        'hl1n18': '4.458760738372803',
        'hl1n19': '4.003769874572754',
        'hl1n20': '3.7527713775634766',
        'hl1n21': '4.09102725982666',
        'hl1n22': '3.759274959564209',
        'hl1n23': '3.975064992904663',
        'hl1n24': '4.317643165588379'
    },
    'hl0n5': {
        'hl1n0': '6.081572532653809',
        'hl1n1': '6.295105457305908',
        'hl1n2': '5.86236572265625',
        'hl1n3': '6.195971965789795',
        'hl1n4': '5.964634418487549',
        'hl1n5': '5.938378810882568',
        'hl1n6': '6.406986236572266',
        'hl1n7': '6.109573841094971',
        'hl1n8': '6.495389461517334',
        'hl1n9': '6.3642048835754395',
        'hl1n10': '6.361320495605469',
        'hl1n11': '6.411458969116211',
        'hl1n12': '6.182220935821533',
        'hl1n13': '6.266909599304199',
        'hl1n14': '6.637297630310059',
        'hl1n15': '6.021630764007568',
        'hl1n16': '6.142656326293945',
        'hl1n17': '6.303307056427002',
        'hl1n18': '5.818584442138672',
        'hl1n19': '6.286212921142578',
        'hl1n20': '5.813422679901123',
        'hl1n21': '6.0800275802612305',
        'hl1n22': '6.038843631744385',
        'hl1n23': '5.836370468139648',
        'hl1n24': '6.615612506866455'
    },
    'hl0n6': {
        'hl1n0': '7.113581657409668',
        'hl1n1': '6.825526714324951',
        'hl1n2': '6.749797821044922',
        'hl1n3': '6.9856438636779785',
        'hl1n4': '6.750903129577637',
        'hl1n5': '7.258749008178711',
        'hl1n6': '7.246515274047852',
        'hl1n7': '7.038250923156738',
        'hl1n8': '6.710267066955566',
        'hl1n9': '7.05867862701416',
        'hl1n10': '7.079568386077881',
        'hl1n11': '7.384864807128906',
        'hl1n12': '7.157761573791504',
        'hl1n13': '6.694675445556641',
        'hl1n14': '7.184409141540527',
        'hl1n15': '6.7659592628479',
        'hl1n16': '6.474432945251465',
        'hl1n17': '7.150697708129883',
        'hl1n18': '6.8418049812316895',
        'hl1n19': '6.9954986572265625',
        'hl1n20': '6.841829299926758',
        'hl1n21': '6.870686054229736',
        'hl1n22': '6.855858325958252',
        'hl1n23': '6.864416122436523',
        'hl1n24': '7.301379203796387'
    },
    'hl0n7': {
        'hl1n0': '5.214013576507568',
        'hl1n1': '4.676849842071533',
        'hl1n2': '5.044111251831055',
        'hl1n3': '5.145365238189697',
        'hl1n4': '4.727697849273682',
        'hl1n5': '4.523938179016113',
        'hl1n6': '4.743663311004639',
        'hl1n7': '5.093776702880859',
        'hl1n8': '4.456836700439453',
        'hl1n9': '4.669329643249512',
        'hl1n10': '4.433446407318115',
        'hl1n11': '5.105783462524414',
        'hl1n12': '5.186365604400635',
        'hl1n13': '4.87766695022583',
        'hl1n14': '5.264307975769043',
        'hl1n15': '5.2975664138793945',
        'hl1n16': '4.996981620788574',
        'hl1n17': '4.918442249298096',
        'hl1n18': '4.441928386688232',
        'hl1n19': '5.033686637878418',
        'hl1n20': '4.51932430267334',
        'hl1n21': '4.3827643394470215',
        'hl1n22': '4.468034744262695',
        'hl1n23': '4.901153087615967',
        'hl1n24': '5.147150993347168'
    },
    'hl1n0': {
        'hl2n0': '6.000449180603027',
        'hl2n1': '5.811243534088135',
        'hl2n2': '-0.19770419597625732',
        'hl2n3': '0.3084714114665985',
        'hl2n4': '6.280590534210205',
        'hl2n5': '-0.3130808174610138',
        'hl2n6': '5.688989639282227',
        'hl2n7': '0.07534044235944748',
        'hl2n8': '5.440340042114258',
        'hl2n9': '6.141881942749023'
    },
    'hl1n1': {
        'hl2n0': '5.601433277130127',
        'hl2n1': '5.9559431076049805',
        'hl2n2': '0.35283344984054565',
        'hl2n3': '-0.2756234109401703',
        'hl2n4': '6.063529968261719',
        'hl2n5': '-0.33646106719970703',
        'hl2n6': '5.616260051727295',
        'hl2n7': '0.30905359983444214',
        'hl2n8': '5.676729679107666',
        'hl2n9': '5.574977397918701'
    },
    'hl1n2': {
        'hl2n0': '6.027472496032715',
        'hl2n1': '5.830715656280518',
        'hl2n2': '0.052814994007349014',
        'hl2n3': '-0.40089333057403564',
        'hl2n4': '5.465182304382324',
        'hl2n5': '-0.24362939596176147',
        'hl2n6': '5.68941068649292',
        'hl2n7': '-0.4209383428096771',
        'hl2n8': '5.573482036590576',
        'hl2n9': '5.893630504608154'
    },
    'hl1n3': {
        'hl2n0': '6.2117509841918945',
        'hl2n1': '5.400324821472168',
        'hl2n2': '-0.1564902514219284',
        'hl2n3': '-0.4172798991203308',
        'hl2n4': '5.863358974456787',
        'hl2n5': '-0.2768084406852722',
        'hl2n6': '5.447022914886475',
        'hl2n7': '0.23308423161506653',
        'hl2n8': '5.348555564880371',
        'hl2n9': '6.126533031463623'
    },
    'hl1n4': {
        'hl2n0': '5.802602291107178',
        'hl2n1': '6.218750476837158',
        'hl2n2': '-0.1610722541809082',
        'hl2n3': '0.26283055543899536',
        'hl2n4': '5.6670308113098145',
        'hl2n5': '0.05790078639984131',
        'hl2n6': '5.838947296142578',
        'hl2n7': '0.19934801757335663',
        'hl2n8': '5.725589752197266',
        'hl2n9': '5.926657676696777'
    },
    'hl1n5': {
        'hl2n0': '5.668773651123047',
        'hl2n1': '5.318731307983398',
        'hl2n2': '0.3193906247615814',
        'hl2n3': '0.04112483933568001',
        'hl2n4': '5.444864273071289',
        'hl2n5': '-0.4120914936065674',
        'hl2n6': '5.594430923461914',
        'hl2n7': '0.12075881659984589',
        'hl2n8': '5.750746250152588',
        'hl2n9': '6.097016334533691'
    },
    'hl1n6': {
        'hl2n0': '5.834757328033447',
        'hl2n1': '6.015623092651367',
        'hl2n2': '-0.33804425597190857',
        'hl2n3': '-0.05146416649222374',
        'hl2n4': '5.6275739669799805',
        'hl2n5': '-0.23585422337055206',
        'hl2n6': '5.757120132446289',
        'hl2n7': '-0.39662545919418335',
        'hl2n8': '6.040685176849365',
        'hl2n9': '5.776689529418945'
    },
    'hl1n7': {
        'hl2n0': '5.799858570098877',
        'hl2n1': '5.618692398071289',
        'hl2n2': '0.1401330679655075',
        'hl2n3': '-0.18286748230457306',
        'hl2n4': '6.098344326019287',
        'hl2n5': '-0.12026502937078476',
        'hl2n6': '5.799036026000977',
        'hl2n7': '0.10734345018863678',
        'hl2n8': '5.231446266174316',
        'hl2n9': '5.9276533126831055'
    },
    'hl1n8': {
        'hl2n0': '5.798445224761963',
        'hl2n1': '5.8456130027771',
        'hl2n2': '0.2058878093957901',
        'hl2n3': '0.19682209193706512',
        'hl2n4': '5.578334331512451',
        'hl2n5': '-0.21868611872196198',
        'hl2n6': '5.355103969573975',
        'hl2n7': '-0.08183490484952927',
        'hl2n8': '5.887269496917725',
        'hl2n9': '5.480276107788086'
    },
    'hl1n9': {
        'hl2n0': '5.564273834228516',
        'hl2n1': '6.044132232666016',
        'hl2n2': '-0.44956350326538086',
        'hl2n3': '-0.3271154463291168',
        'hl2n4': '6.02701997756958',
        'hl2n5': '0.35673949122428894',
        'hl2n6': '5.515003204345703',
        'hl2n7': '0.005755146499723196',
        'hl2n8': '5.124919891357422',
        'hl2n9': '6.027717113494873'
    },
    'hl1n10': {
        'hl2n0': '6.013672828674316',
        'hl2n1': '5.513575077056885',
        'hl2n2': '-0.24887068569660187',
        'hl2n3': '-0.09675474464893341',
        'hl2n4': '5.920039653778076',
        'hl2n5': '-0.3189733922481537',
        'hl2n6': '5.92581033706665',
        'hl2n7': '0.2404550462961197',
        'hl2n8': '5.108793258666992',
        'hl2n9': '5.702240467071533'
    },
    'hl1n11': {
        'hl2n0': '5.626531600952148',
        'hl2n1': '6.075304985046387',
        'hl2n2': '0.32012975215911865',
        'hl2n3': '-0.1187189444899559',
        'hl2n4': '5.488193988800049',
        'hl2n5': '-0.2695629894733429',
        'hl2n6': '5.706921577453613',
        'hl2n7': '-0.01139084529131651',
        'hl2n8': '5.517491817474365',
        'hl2n9': '5.789714813232422'
    },
    'hl1n12': {
        'hl2n0': '5.769123077392578',
        'hl2n1': '6.147134780883789',
        'hl2n2': '-0.20994503796100616',
        'hl2n3': '-0.15055933594703674',
        'hl2n4': '6.255430698394775',
        'hl2n5': '-0.3471902310848236',
        'hl2n6': '5.52802848815918',
        'hl2n7': '-0.028450965881347656',
        'hl2n8': '5.543689727783203',
        'hl2n9': '6.376387119293213'
    },
    'hl1n13': {
        'hl2n0': '5.795427322387695',
        'hl2n1': '5.847070693969727',
        'hl2n2': '0.13453027606010437',
        'hl2n3': '-0.22435356676578522',
        'hl2n4': '5.744321346282959',
        'hl2n5': '-0.03509828448295593',
        'hl2n6': '5.956820964813232',
        'hl2n7': '-0.4013110399246216',
        'hl2n8': '5.897938251495361',
        'hl2n9': '5.909621238708496'
    },
    'hl1n14': {
        'hl2n0': '5.522014141082764',
        'hl2n1': '6.135678768157959',
        'hl2n2': '0.20999044179916382',
        'hl2n3': '-0.21209441125392914',
        'hl2n4': '5.884896755218506',
        'hl2n5': '0.16111060976982117',
        'hl2n6': '5.853396892547607',
        'hl2n7': '0.3344040513038635',
        'hl2n8': '5.378990650177002',
        'hl2n9': '6.234222888946533'
    },
    'hl1n15': {
        'hl2n0': '5.774571895599365',
        'hl2n1': '5.265669345855713',
        'hl2n2': '-0.4065327048301697',
        'hl2n3': '-0.22100909054279327',
        'hl2n4': '5.29531717300415',
        'hl2n5': '0.24212335050106049',
        'hl2n6': '5.5713210105896',
        'hl2n7': '0.219936802983284',
        'hl2n8': '5.4527482986450195',
        'hl2n9': '5.993016719818115'
    },
    'hl1n16': {
        'hl2n0': '6.19078254699707',
        'hl2n1': '5.5579986572265625',
        'hl2n2': '-0.36100178956985474',
        'hl2n3': '0.3766549825668335',
        'hl2n4': '5.967555999755859',
        'hl2n5': '0.3198370337486267',
        'hl2n6': '5.356962203979492',
        'hl2n7': '0.06086554005742073',
        'hl2n8': '5.393331050872803',
        'hl2n9': '5.598042011260986'
    },
    'hl1n17': {
        'hl2n0': '5.965753555297852',
        'hl2n1': '5.835743427276611',
        'hl2n2': '-0.012837335467338562',
        'hl2n3': '0.23828907310962677',
        'hl2n4': '5.2913641929626465',
        'hl2n5': '0.1044221743941307',
        'hl2n6': '5.796876907348633',
        'hl2n7': '0.035544171929359436',
        'hl2n8': '5.589445114135742',
        'hl2n9': '6.06375789642334'
    },
    'hl1n18': {
        'hl2n0': '5.730203151702881',
        'hl2n1': '5.404701232910156',
        'hl2n2': '-0.4376900792121887',
        'hl2n3': '0.3288637697696686',
        'hl2n4': '6.074588298797607',
        'hl2n5': '0.11017391085624695',
        'hl2n6': '5.690276145935059',
        'hl2n7': '-0.3255769908428192',
        'hl2n8': '5.46752405166626',
        'hl2n9': '5.511534690856934'
    },
    'hl1n19': {
        'hl2n0': '5.702664852142334',
        'hl2n1': '6.025961875915527',
        'hl2n2': '-0.0343628004193306',
        'hl2n3': '-0.08956869691610336',
        'hl2n4': '5.919732570648193',
        'hl2n5': '-0.059078969061374664',
        'hl2n6': '6.094509601593018',
        'hl2n7': '-0.2937847375869751',
        'hl2n8': '5.905513286590576',
        'hl2n9': '5.758819103240967'
    },
    'hl1n20': {
        'hl2n0': '5.516214847564697',
        'hl2n1': '5.273271560668945',
        'hl2n2': '0.20604103803634644',
        'hl2n3': '-0.030304910615086555',
        'hl2n4': '5.560956954956055',
        'hl2n5': '0.05965522304177284',
        'hl2n6': '5.598865509033203',
        'hl2n7': '-0.024875091388821602',
        'hl2n8': '5.834478855133057',
        'hl2n9': '5.371649265289307'
    },
    'hl1n21': {
        'hl2n0': '6.154098987579346',
        'hl2n1': '5.722245216369629',
        'hl2n2': '0.03906470164656639',
        'hl2n3': '-0.35830333828926086',
        'hl2n4': '6.166950225830078',
        'hl2n5': '-0.26759958267211914',
        'hl2n6': '5.45729923248291',
        'hl2n7': '-0.1558355987071991',
        'hl2n8': '5.907253265380859',
        'hl2n9': '6.016402244567871'
    },
    'hl1n22': {
        'hl2n0': '6.2118306159973145',
        'hl2n1': '6.15168571472168',
        'hl2n2': '0.10640759021043777',
        'hl2n3': '0.10598719865083694',
        'hl2n4': '5.97401762008667',
        'hl2n5': '0.31231218576431274',
        'hl2n6': '5.465885162353516',
        'hl2n7': '-0.2012670636177063',
        'hl2n8': '5.479915618896484',
        'hl2n9': '5.488998889923096'
    },
    'hl1n23': {
        'hl2n0': '5.977294921875',
        'hl2n1': '5.5201640129089355',
        'hl2n2': '-0.20402641594409943',
        'hl2n3': '-0.05771653726696968',
        'hl2n4': '5.5712714195251465',
        'hl2n5': '-0.4320337772369385',
        'hl2n6': '5.93337345123291',
        'hl2n7': '-0.37542271614074707',
        'hl2n8': '5.206535339355469',
        'hl2n9': '5.931999206542969'
    },
    'hl1n24': {
        'hl2n0': '6.19212007522583',
        'hl2n1': '5.515852451324463',
        'hl2n2': '-0.38143694400787354',
        'hl2n3': '0.20691300928592682',
        'hl2n4': '5.740112781524658',
        'hl2n5': '-0.07349052280187607',
        'hl2n6': '5.831645965576172',
        'hl2n7': '-0.2995266020298004',
        'hl2n8': '5.339826583862305',
        'hl2n9': '5.4853715896606445'
    },
    'hl2n0': {
        'hl3n0': '5.962421894073486'
    },
    'hl2n1': {
        'hl3n0': '6.157137870788574'
    },
    'hl2n2': {
        'hl3n0': '-0.35607463121414185'
    },
    'hl2n3': {
        'hl3n0': '-0.3668365180492401'
    },
    'hl2n4': {
        'hl3n0': '6.065285682678223'
    },
    'hl2n5': {
        'hl3n0': '-0.023579725995659828'
    },
    'hl2n6': {
        'hl3n0': '6.324278354644775'
    },
    'hl2n7': {
        'hl3n0': '-0.11444011330604553'
    },
    'hl2n8': {
        'hl3n0': '6.703828811645508'
    },
    'hl2n9': {
        'hl3n0': '5.968878746032715'
    }
}

def update_element_values(elements, inputs):
    layers = []

    # Compute first layer values with inputs
    # Make algorithm to automatically generate based on previous layer data

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