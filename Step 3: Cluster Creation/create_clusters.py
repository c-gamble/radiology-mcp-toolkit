import sys
sys.path.append("./utils/")

from iou import calculate_iou
from typing import List
import copy
from collections import defaultdict
from tqdm.auto import tqdm

# Define IoU Threshold. This is the value found to be optimal for our task, but we encourage you to test values between 0 and 1 for your dataset/task.
IOU_THRESHOLD: float = 0.6

# Read the post-suppression predictions from Step 2.
with open("./Step 2: Class-wise NMS/test_predictions_retained.txt", 'r') as f:
    lines: str = f.read().split('\n\n')[1]
    exec(lines) # this will define dictionary of predictions called test_predictions_retained


test_predictions_clustered: dict = {}
"""
test_predictions_clustered: {
    'instance identifier': {
        'n_clusters': int,
        'clusters': {
            0: {
                'hemorrhage type': {
                    'HNU': HNU,
                    'bounding_box': bounding_box (yolo format)
                },
                'hemorrhage type': {
                    'HNU': HNU,
                    'bounding_box': bounding_box (yolo format)
                },
                ...
            },
            1: ...
        }
    }
}
"""

# Generate all possible clusters.
for label_file, hemorrhage_types in tqdm(test_predictions_retained.items()):

    # Define an empty dictionary to store clusters and start a counter for the number of clusters.
    clusters: dict = defaultdict(dict)
    n_clusters: int = 0

    for h_type, details in hemorrhage_types.items():

        # Pick a hemorrhage type and get its HNU and bounding box.
        HNU: float = float(details['HNU'])
        bbox: str = details['bounding_box']

        # If the HNU is -1, then there is no prediction for this hemorrhage type.
        if HNU == -1: continue

        # We now create a new cluster with an ID equal to the current number of clusters. We then add the hemorrhage we picked to the cluster.
        clusters[n_clusters][h_type] = details

        for other_htype, other_details in hemorrhage_types.items():
            if h_type == other_htype: continue

            # Pick another hemorrhage type and get its HNU and bounding box.
            other_HNU: float = float(other_details['HNU'])
            other_bbox: str = other_details['bounding_box']

            if other_HNU == -1: continue

            iou: float = calculate_iou(bbox, other_bbox)

            # If the IOU is greater than the threshold, then we add the hemorrhage to the cluster.
            if iou > IOU_THRESHOLD:   
                clusters[n_clusters][other_htype] = other_details
        
        n_clusters += 1
    test_predictions_clustered[label_file] = {'n_clusters': n_clusters, 'clusters': clusters}

# Remove duplicate clusters.
for label_file, cluster_details in tqdm(test_predictions_clustered.items()):
    
    # Get the clusters and the number of clusters.
    clusters: dict = cluster_details['clusters']
    n_clusters: int = cluster_details['n_clusters']

    # Define a set to store the unique hemorrhage types for each cluster.
    htype_tups: set = set()

    # Make a copy of the clusters dictionary for iteration.
    cluster_copy: dict = copy.deepcopy(clusters)

    for cluster_id, cluster in cluster_copy.items():
        
        # If a given cluster has the exact same hemorrhage types as another cluster, then we assume they are the same cluster. 
        htype_tup: tuple = tuple(sorted(cluster.keys()))
        if htype_tup in htype_tups:
            del clusters[cluster_id]
            n_clusters -= 1
        else:
            htype_tups.add(htype_tup)

    # Reset cluster IDs.
    new_clusters: dict = {}
    for i, cluster in enumerate(clusters.values()):
        new_clusters[i] = cluster
    clusters: dict = new_clusters
    test_predictions_clustered[label_file] = {'n_clusters': n_clusters, 'clusters': clusters}

# Save the retained predictions (for tutorial purposes only).
with open("./Step 3: Cluster Creation/test_predictions_clustered.txt", 'w') as f:
    f.write("This .txt file is for tutorial purposes only. Please avoid saving your dictionary output by combining multiple steps into a single .py file. If you must save your dictionary output, please use a .pkl file or other efficient storage method.\n\n")
    f.write("test_predictions_clustered = {\n")
    for instance in test_predictions_clustered:
        f.write(f"    '{instance}': {{\n")
        f.write(f"        'n_clusters': {test_predictions_clustered[instance]['n_clusters']},\n")
        f.write(f"        'clusters': {{\n")
        for cluster_id in test_predictions_clustered[instance]['clusters']:
            f.write(f"            {cluster_id}: {{\n")
            for h_type in test_predictions_clustered[instance]['clusters'][cluster_id]:
                f.write(f"                '{h_type}': {{\n")
                for key in test_predictions_clustered[instance]['clusters'][cluster_id][h_type]:
                    f.write(f"                    '{key}': '{test_predictions_clustered[instance]['clusters'][cluster_id][h_type][key]}',\n")
                f.write("                },\n")
            f.write("            },\n")
        f.write("        },\n")
        f.write("    },\n")
    f.write("}")
    