import sys
sys.path.append("./utils/")

from iou import calculate_iou
from typing import List
from tqdm.auto import tqdm

# Define IoU Threshold. This is the value found to be optimal for our task, but we encourage you to test values between 0 and 1 for your dataset/task.
IOU_THRESHOLD: float = 0.25

int_to_label: dict = {0: 'IPH', 1: 'IVH', 2: 'SDH', 3: 'EDH', 4: 'SAH'}
h_types: List[str] = list(int_to_label.values())

# Read the clusters created in Step 3.
with open("./Step 3: Cluster Creation/test_predictions_clustered.txt", 'r') as f:
    lines: str = f.read().split('\n\n')[1]
    exec(lines) # this will define dictionary of predictions called test_predictions_clustered

# We also need the original 10,000 predictions and HNUs from Step 1
with open("./Step 1: Raw Predictions/test_HNUs.txt", 'r') as f:
    lines: str = f.read().split('\n\n')[1]
    exec(lines) # this will define dictionary of HNUs called test_HNUs

test_clusters_filled: dict = {}
"""
test_clusters_filled: {
    'instance identifier': {
        'n_clusters': int,
        'clusters': {
            0: {
                'IPH': {
                    'HNU': HNU,
                    'bounding_box': bounding_box (yolo format)
                },
                'IVH': ...,
                'SDH': ...,
                'EDH': ...,
                'SAH': ...
            },
            1: ...
"""
for label_file, cluster_details in tqdm(test_predictions_clustered.items()):

    # Pick a sample.
    clusters: dict = cluster_details['clusters']
    n_clusters: int = cluster_details['n_clusters']

    # Define an empty dictionary to store clusters and start a counter for the number of clusters.
    test_clusters_filled[label_file] = {'n_clusters': n_clusters, 'clusters': {}}
    for cluster_id, cluster in clusters.items():

        # Pick a cluster and find the maximum confidence prediction.
        max_confidence_existing: float = -1
        max_confidence_bbox: str = ''
        for h_type_existing in cluster:
            if float(cluster[h_type_existing]['HNU']) > max_confidence_existing:
                max_confidence_existing = float(cluster[h_type_existing]['HNU'])
                max_confidence_bbox = cluster[h_type_existing]['bounding_box']

        # Define an empty dictionary to store the filled cluster.
        test_clusters_filled[label_file]['clusters'][cluster_id]: dict = {}

        for h_type in h_types:
            # If the hemorrhage type is in the cluster, then we just use the information we already have.
            if h_type in cluster:
                test_clusters_filled[label_file]['clusters'][cluster_id][h_type] = cluster[h_type]
            else:  
                # If the hemorrhage type is not in the cluster, then we need to fill it with the prediction that has the highest HNU and an IoU > IOU_THRESHOLD.      
                test_clusters_filled[label_file]['clusters'][cluster_id][h_type] = {'HNU': -1, 'bounding_box': ''}
                for HNU, bbox in test_HNUs[label_file][h_type]:
                    if HNU == -1: continue
                    iou = calculate_iou(bbox, max_confidence_bbox)
                    if iou > IOU_THRESHOLD:
                        # Cluster-wise NMS (Step 5) happens here. It's much simpler than creating a new file for Step 5.
                        if HNU > test_clusters_filled[label_file]['clusters'][cluster_id][h_type]['HNU']:
                            test_clusters_filled[label_file]['clusters'][cluster_id][h_type]['HNU'] = HNU
                            test_clusters_filled[label_file]['clusters'][cluster_id][h_type]['bounding_box'] = bbox

# Save the filled clusters (for tutorial purposes only).
with open("./Step 4 and Step 5: Cluster Filling/test_clusters_filled.txt", 'w') as f:
    f.write("This .txt file is for tutorial purposes only. Please avoid saving your dictionary output by combining multiple steps into a single .py file. If you must save your dictionary output, please use a .pkl file or other efficient storage method.\n\n")
    f.write("test_clusters_filled = {\n")
    for instance in test_clusters_filled:
        f.write(f"    '{instance}': {{\n")
        f.write(f"        'n_clusters': {test_clusters_filled[instance]['n_clusters']},\n")
        f.write(f"        'clusters': {{\n")
        for cluster_id in test_clusters_filled[instance]['clusters']:
            f.write(f"            {cluster_id}: {{\n")
            for h_type in test_clusters_filled[instance]['clusters'][cluster_id]:
                f.write(f"                '{h_type}': {{\n")
                f.write(f"                    'HNU': {test_clusters_filled[instance]['clusters'][cluster_id][h_type]['HNU']},\n")
                f.write(f"                    'bounding_box': '{test_clusters_filled[instance]['clusters'][cluster_id][h_type]['bounding_box']}',\n")
                f.write(f"                }},\n")
            f.write(f"            }},\n")
        f.write(f"        }},\n")
        f.write(f"    }},\n")
    f.write(f"}}")