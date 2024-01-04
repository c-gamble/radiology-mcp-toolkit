import sys
sys.path.append("./utils/")

from insert_index import find_insert_index
from typing import List
import pandas as pd
from tqdm.auto import tqdm

# Define IoU Threshold. This is the value found to be optimal for our task, but we encourage you to test values between 0 and 1 for your dataset/task.
CONFORMAL_SCORE_THRESHOLD: float = 0.95

int_to_label: dict = {0: 'IPH', 1: 'IVH', 2: 'SDH', 3: 'EDH', 4: 'SAH'}
h_types: List[str] = list(int_to_label.values())

# Read the filled clusters from Steps 4 and 5.
with open("./Step 4 and Step 5: Cluster Filling/test_clusters_filled.txt", 'r') as f:
    lines: str = f.read().split('\n\n')[1]
    exec(lines) # this will define dictionary of predictions called test_clusters_filled


# We will need the calibration HNUs to calculate the conformal score for each test instance in this Step.
calibration_HNUs_df: pd.DataFrame = pd.read_csv('./Step 0: Calibration/calibration_HNUs.csv')
calibration_HNUs_dict: dict = {
    'IPH': {
        'presence': [],
        'absence': [],
    },
    'IVH': {
        'presence': [],
        'absence': [],
    },
    'SDH': {
        'presence': [],
        'absence': [],
    },
    'EDH': {
        'presence': [],
        'absence': [],
    },
    'SAH': {
        'presence': [],
        'absence': [],
    }
}

for _, row in tqdm(list(calibration_HNUs_df.iterrows())):
    for h_type in h_types: 
        for presence in ['Presence', 'Absence']:
            calibration_HNUs_dict[h_type][presence.lower()].append(row[f'{h_type} {presence}'])
for h_type in h_types:
    for presence in ['presence', 'absence']:
        calibration_HNUs_dict[h_type][presence] = sorted(calibration_HNUs_dict[h_type][presence])

# Calculate the conformal score for each prediction in each cluster.
filled_clusters_with_conformal_scores: dict = {}
for label_file, cluster_details in tqdm(test_clusters_filled.items()):
    clusters: dict = cluster_details['clusters']
    n_clusters: int = cluster_details['n_clusters']

    # Define an empty dictionary to store clusters.
    filled_clusters_with_conformal_scores[label_file] = {'n_clusters': n_clusters, 'clusters': {}}
    for cluster_id, cluster in clusters.items():

        # Pick a cluster and define an empty dictionary to store the filled cluster.
        filled_clusters_with_conformal_scores[label_file]['clusters'][cluster_id] = {}
        for h_type in h_types:

            # Ignore the hemorrhage type if it is not present in the cluster.
            if cluster[h_type]['HNU'] == -1:
                filled_clusters_with_conformal_scores[label_file]['clusters'][cluster_id][h_type] = {'HNU': -1, 'bounding_box': '', 'absence_conformal_score': -1, 'presence_conformal_score': -1}
            
            # Otherwise calculate the conformal score for the hemorrhage type's absence and presence within the cluster.
            else:
                score = cluster[h_type]['HNU']
                presence_conformal_score = find_insert_index(calibration_HNUs_dict[h_type]['presence'], score) / len(calibration_HNUs_dict[h_type]['presence'])
                absence_conformal_score = find_insert_index(calibration_HNUs_dict[h_type]['absence'], score) / len(calibration_HNUs_dict[h_type]['absence'])
                filled_clusters_with_conformal_scores[label_file]['clusters'][cluster_id][h_type] = {'HNU': score, 'bounding_box': cluster[h_type]['bounding_box'], 'absence_conformal_score': absence_conformal_score, 'presence_conformal_score': presence_conformal_score}

# Generate the prediction set for each test instance.
test_prediction_sets = {}
for label_file, cluster_details in tqdm(filled_clusters_with_conformal_scores.items()):
    clusters = cluster_details['clusters']
    n_clusters = cluster_details['n_clusters']

    # Define an empty prediction set for each test instance.
    test_prediction_sets[label_file] = {'n_clusters': n_clusters, 'clusters': {}, 'prediction_set': set()}
    for cluster_id, cluster in clusters.items():
        test_prediction_sets[label_file]['clusters'][cluster_id] = {}
        for h_type in h_types:
            # Ignore the hemorrhage type if it is not present in the cluster.
            if cluster[h_type]['HNU'] == -1:
                test_prediction_sets[label_file]['clusters'][cluster_id][h_type] = {'HNU': -1, 'bounding_box': '', 'absence_conformal_score': -1, 'presence_conformal_score': -1}
            
            # Otherwise calculate compare the conformal score for the hemorrhage type's absence and presence to the threshold.
            else:
                score = cluster[h_type]['HNU']
                presence_conformal_score = cluster[h_type]['presence_conformal_score']
                absence_conformal_score = cluster[h_type]['absence_conformal_score']
                
                if presence_conformal_score > CONFORMAL_SCORE_THRESHOLD:
                    x, y, w, h = [float(x)*512 for x in cluster[h_type]['bounding_box'].split(' ')]
                    test_prediction_sets[label_file]['prediction_set'].add(f'{h_type} (C={score:.2f}) centered at ({x:.2f}, {y:.2f}) with width {w:.2f} and height {h:.2f}')
                if absence_conformal_score > CONFORMAL_SCORE_THRESHOLD:
                    test_prediction_sets[label_file]['prediction_set'].add(f'No {h_type} (C={1-score:.2f}) in cluster {cluster_id}')
                
                test_prediction_sets[label_file]['clusters'][cluster_id][h_type] = {'HNU': score, 'bounding_box': cluster[h_type]['bounding_box'], 'absence_conformal_score': absence_conformal_score, 'presence_conformal_score': presence_conformal_score}

# Save the prediction sets (for tutorial purposes only).
with open("./Step 6: Prediction Sets/test_prediction_sets.txt", 'w') as f:
    f.write("This .txt file is for tutorial purposes only. Please avoid saving your dictionary output by combining multiple steps into a single .py file. If you must save your dictionary output, please use a .pkl file or other efficient storage method.\n\n")
    f.write("test_prediction_sets = {\n")
    for instance in test_prediction_sets:
        f.write(f"    '{instance}': {{\n")
        f.write(f"        'n_clusters': {test_prediction_sets[instance]['n_clusters']},\n")
        f.write(f"        'clusters': {{\n")
        for cluster_id in test_prediction_sets[instance]['clusters']:
            f.write(f"            {cluster_id}: {{\n")
            for h_type in test_prediction_sets[instance]['clusters'][cluster_id]:
                f.write(f"                '{h_type}': {{\n")
                for key in test_prediction_sets[instance]['clusters'][cluster_id][h_type]:
                    f.write(f"                    '{key}': '{test_prediction_sets[instance]['clusters'][cluster_id][h_type][key]}',\n")
                f.write("                },\n")
            f.write("            },\n")
        f.write("        },\n")
        f.write(f"        'prediction_set': {test_prediction_sets[instance]['prediction_set']},\n")
        f.write("    },\n")
    f.write("}")

