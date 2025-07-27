import glob
import os
import json
import spacy
from string import punctuation
import statistics
import matplotlib.pyplot as plt


def get_filenames(folder_path): 
    """ 
    Retrieves all JSON file names in a specified directory. 
    Returns: Set of file names (without path).
    """
    return set(os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*.json')))


def lemmatize_clean(text):
    """ 
    Lemmatizes the text and removes stop words and punctuation using the spaCy pipeline. 
    Returns: List of lemmatized and cleaned words.
    """
    text = text.lower()
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_stop and token.text not in punctuation]
    return lemmatized


# Retrieve JSON files from the MDACE data for Inpatient and Profee
inpatient_files = get_filenames('with_text/gold/Inpatient/ICD-10/1.0')
profee_files = get_filenames('with_text/gold/Profee/ICD-10/1.0')

# Combine all files
# Since the code overlap between Inpatient and Profee is minor, we combine all for this analysis of overlap computation
all_files = glob.glob(os.path.join('with_text/gold/Inpatient/ICD-10/1.0', '*.json'))
all_files += glob.glob(os.path.join('with_text/gold/Profee/ICD-10/1.0', '*.json'))

# Load the spaCy model for the pipeline
nlp = spacy.load('en_core_web_sm')

result = {}  # dictionary with code description as key and evidence as items
median_list = []  # list to store overlap scores for each code (description)

# Process all files, extract evidence for each code description
for f in all_files: 
    with open(f, 'r') as one_file: 
        data = json.load(one_file)
    for n in data['notes']:
        # extract explanations according to code description assuming that code descriptions are unique
        for annotation in n['annotations']:
            description = annotation['description']
            evidence = annotation['covered_text']
            if description not in result:
                result[description] = []
            result[description].append(evidence)

# Compute overlap between code descriptions and asociated evidence
for description, evidence_list in result.items(): 
    lem_description = lemmatize_clean(description)  # lemmatize code descriptions
    overlap_list = []
    for e in evidence_list: 
        lem_evidence = lemmatize_clean(e)  # lemmatize evidence span
        code_description_intersection = set(lem_description).intersection(lem_evidence)  # compute intersection of words
        if code_description_intersection:
            overlap = len(code_description_intersection)/len(lem_description)  # normalize 
            overlap_list.append(overlap)
        else: 
            overlap_list.append(0)
    if not overlap_list:
        median_value = 0
    else: 
        median_value = statistics.median(overlap_list)  # median value for description
    median_list.append(median_value)  

fig, ax = plt.subplots(figsize=(6, 6))
fig.set_size_inches(w=3.2, h=1.8)

color_palette = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
plt.hist(median_list, bins=20, range=(0, 1), color=color_palette[1], edgecolor='black')

plt.ylabel('ICD code count')
plt.xlabel('Median overlap')

plt.tight_layout()
plt.show()
