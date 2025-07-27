import glob
import os
import json
from similarity_methods import *
import matplotlib.pyplot as plt


def get_xlist(annotations):
    """ 
    Retrieves evidence from MDACE annotations. 
    Arg: List of annotation dictionaries.
    Returns: List evidence.
    """
    word_list = [a['covered_text'] for a in annotations]
    return word_list


def is_subset(list1, list2):
    """ 
    Checks if list1 is a subset of list2.
    Args:
        list1 (list): First list.
        list2 (list): Second list.
    Returns: bool - True if list1 is a subset of list2, False otherwise.
    """
    set1 = set(list1)
    set2 = set(list2)
    return set1.issubset(set2)


def get_filenames(folder_path): 
    """ 
    Retrieves all JSON file names in a specified directory.
    Args:
        folder_path (str): Path to the directory containing JSON files.
    Returns: Set of file names (without path).
    """
    return set(os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*.json')))


def get_codes(json_data):
    """ 
    Extracts codes from annotations in JSON data.
    Args: json_data (dict): JSON data containing annotations.
    Returns: List of codes from the annotations.
    """
    return [annotation['code'] for annotation in json_data['annotations']]


def get_annotations(note, common_codes):
    """ 
    Filters annotations in a note based on common codes.
    Args:
        note (dict): Note containing annotations.
        common_codes (set): Set of common codes to filter by.
    Returns: List of filtered annotations.
    """
    filtered_annotations = []
    for annotation in note['annotations']:
        if annotation['code'] in common_codes:
            filtered_annotations.append(annotation)
    return filtered_annotations


color_palette = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']

inpatient_files = get_filenames('with_text/gold/Inpatient/ICD-10/1.0')
profee_files = get_filenames('with_text/gold/Profee/ICD-10/1.0')
common_filenames = inpatient_files.intersection(profee_files)

all_common_note_ids = 0
inpatient_discharge_list = []
profee_discharge_list = []
inpatient_physician_list = []
profee_physician_list = []

# Process each common file from Inpatient and Profee and extract position information in lists
for i in common_filenames: 

    with open(f'with_text/gold/Inpatient/ICD-10/1.0/{i}', 'r') as file_i: 
        inpatient_file = json.load(file_i)
    with open(f'with_text/gold/Profee/ICD-10/1.0/{i}', 'r') as file_p: 
        profee_file = json.load(file_p)    

    # Extract note IDs from both files
    inpatient_note_ids = [n['note_id'] for n in inpatient_file['notes']]
    profee_note_ids = [n['note_id'] for n in profee_file['notes']]

    # Find common note IDs
    common_note_ids = list(set(profee_note_ids) & set(inpatient_note_ids))
    all_common_note_ids += len(common_note_ids)

    for j in common_note_ids:
        # Retrieve common notes
        inpatient_note = next((note for note in inpatient_file['notes'] if note['note_id'] == j), None)
        profee_note = next((note for note in profee_file['notes'] if note['note_id'] == j), None)  # intention: retrieve relevant note from Profee based on Inpatient note

        note_len = len(inpatient_note['text'])
        # Filter according to Discharge summaries and Physician notes, can be adjusted for other document types
        if inpatient_note['category'] == "Discharge summary":
            i_d_pos = [a['begin']/note_len for a in inpatient_note['annotations']]
            inpatient_discharge_list += i_d_pos
        if profee_note['category'] == "Discharge summary":
            p_d_pos = [a['begin']/note_len for a in profee_note['annotations']]
            profee_discharge_list += p_d_pos
        
        if inpatient_note['category'] == "Physician":
            i_p_pos = [a['begin']/note_len for a in inpatient_note['annotations']]
            inpatient_physician_list += i_p_pos
        if profee_note['category'] == "Physician":
            p_p_pos = [a['begin']/note_len for a in profee_note['annotations']]
            profee_physician_list += p_p_pos


def plot_position(positions, color): 
    """ 
    Plots a histogram of the positions.
    """
    fig,ax=plt.subplots(figsize=(3,3))
    fig.set_size_inches(w=1.55, h=1.4)
    plt.hist(positions, bins=10, range=(0, 1), edgecolor='black', density=True, color=color)
    plt.ylim(0, 3.1)
    plt.tight_layout()
    plt.show()


plot_position(inpatient_discharge_list, color_palette[0])
plot_position(profee_discharge_list, color_palette[4])
plot_position(inpatient_physician_list, color_palette[0])
plot_position(profee_physician_list, color_palette[4])


