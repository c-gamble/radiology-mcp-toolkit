import os
import numpy as np
import pandas as pd
from typing import List
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm.auto import tqdm
import pickle

int_to_label = {0: 'IPH', 1: 'IVH', 2: 'SDH', 3: 'EDH', 4: 'SAH'}
h_types = list(int_to_label.values())

conformal_scores = defaultdict(dict)
"""
conformal_scores = {
    'label file name': {
        'hemorrhage type': [
            [score, bounding_box (yolo format)],
        ]
    }
"""

for pred in tqdm(os.listdir('/research/projects/Cooper/CQ500_UQ/paper_prep/test_runs/run_1_seed_829/labels/')):
    with open('/research/projects/Cooper/CQ500_UQ/paper_prep/test_runs/run_1_seed_829/labels/' + pred, 'r') as f:
        lines = f.readlines()
        
        predictions_by_class = {'IPH': [], 'IVH': [], 'SDH': [], 'EDH': [], 'SAH': []}
        for line in lines:
            score = float(line.split(' ')[-1].split('\n')[0])
            label_idx = int(float(line.split(' ')[0]))
            bounding_box = " ".join(line.split(' ')[1:5])
            predictions_by_class[int_to_label[label_idx]].append([score, bounding_box])
        
    conformal_scores[pred] = predictions_by_class
print('Completed Step 0: Extracting predictions from raw files')

retained_preds = {}
"""
retained_preds = {
    'label file name': {
        'hemorrhage type': {
            'score': score, (-1 if no bounding box)
            'bounding_box': bounding_box (yolo format) ('' if no bounding box)
        }
    }
"""
for calibration_sample, calibration_dict in tqdm(list(conformal_scores.items())):

    retained_class_bounding_boxes = {}

    for pred_type, score_list in calibration_dict.items():
        max_score = -1
        max_bounding_box = ''
        for bounding_box_score in score_list:
            if bounding_box_score[0] > max_score:
                max_score = bounding_box_score[0]
                max_bounding_box = bounding_box_score[1]
        retained_class_bounding_boxes[pred_type] = {'score': max_score, 'bounding_box': max_bounding_box}
    retained_preds[calibration_sample] = retained_class_bounding_boxes
print('Completed Step 1: Extract max-confidence predictions for each class')


for IOU_THRESHOLD in np.arange(0.0, 1.05, 0.05):
    for P_VALUE_THRESHOLD in np.arange(0.0, 1.05, 0.05):

        def calculate_iou(boxA, boxB, eps=1e-7):
            """
            boxA: 'x y w h'
            boxB: 'x y w h'
            """

            # [centerX, centerY, width, height]
            boxA = [float(n)*512 for n in boxA.split(' ')] 
            boxB = [float(n)*512 for n in boxB.split(' ')]
            
            # [(x_min, y_min), (x_max, y_max)]
            boxA = [(boxA[0] - boxA[2]/2, boxA[1] - boxA[3]/2), (boxA[0] + boxA[2]/2, boxA[1] + boxA[3]/2)]
            boxB = [(boxB[0] - boxB[2]/2, boxB[1] - boxB[3]/2), (boxB[0] + boxB[2]/2, boxB[1] + boxB[3]/2)]

            # Calculate tl and br for intersection
            x_start = max(boxA[0][0], boxB[0][0])
            y_start = max(boxA[0][1], boxB[0][1])
            x_end = min(boxA[1][0], boxB[1][0])
            y_end = min(boxA[1][1], boxB[1][1])

            # Calculate area of intersection
            intersection_area = max(0, x_end - x_start) * max(0, y_end - y_start)

            # Calculate area of both boxes
            boxA_area = (boxA[1][0] - boxA[0][0]) * (boxA[1][1] - boxA[0][1])
            boxB_area = (boxB[1][0] - boxB[0][0]) * (boxB[1][1] - boxB[0][1])

            # Calculate intersection over union
            iou = intersection_area / float(boxA_area + boxB_area - intersection_area + eps)

            return iou

        clustered_predictions = {}
        """
        clustered_predictions: {
            'label file name': {
                'n_clusters': int,
                'clusters': {
                    0: {
                        'hemorrhage type': {
                            'score': score,
                            'bounding_box': bounding_box (yolo format)
                        },
                        'hemorrhage type': {
                            'score': score,
                            'bounding_box': bounding_box (yolo format)
                        },
                        ...
                    },
                    1: ...
                }
            }
        }
        """
        # generate all possible clusters
        for label_file, hemorrhage_types in tqdm(retained_preds.items()):
            clusters = defaultdict(dict)
            n_clusters = 0

            for h_type, details in hemorrhage_types.items():
                score = details['score']
                bbox = details['bounding_box']

                if score == -1: continue
                clusters[n_clusters][h_type] = details

                for other_htype, other_details in hemorrhage_types.items():
                    if h_type == other_htype: continue

                    other_score = other_details['score']
                    other_bbox = other_details['bounding_box']

                    if other_score == -1: continue

                    iou = calculate_iou(bbox, other_bbox)

                    if iou > IOU_THRESHOLD:   
                        clusters[n_clusters][other_htype] = other_details
                
                n_clusters += 1
            clustered_predictions[label_file] = {'n_clusters': n_clusters, 'clusters': clusters}

        # remove duplicate clusters
        for label_file, cluster_details in tqdm(clustered_predictions.items()):
            clusters = cluster_details['clusters']
            n_clusters = cluster_details['n_clusters']

            htype_tups = set()
            cluster_copy = copy.deepcopy(clusters)
            for cluster_id, cluster in cluster_copy.items():
                htype_tup = tuple(sorted(cluster.keys()))
                if htype_tup in htype_tups:
                    del clusters[cluster_id]
                    n_clusters -= 1
                else:
                    htype_tups.add(htype_tup)

            # reset cluster ids
            new_clusters = {}
            for i, cluster in enumerate(clusters.values()):
                new_clusters[i] = cluster
            clusters = new_clusters
            clustered_predictions[label_file] = {'n_clusters': n_clusters, 'clusters': clusters}

        print('Completed Step 2: Clustering predictions')

        filled_clusters = {}
        """
        filled_clusters: {
            'label file name': {
                'n_clusters': int,
                'clusters': {
                    0: {
                        'IPH': {
                            'score': score,
                            'bounding_box': bounding_box (yolo format)
                        },
                        'IVH': ...,
                        'SDH': ...,
                        'EDH': ...,
                        'SAH': ...
                    },
                    1: ...
        """
        for label_file, cluster_details in tqdm(clustered_predictions.items()):
            clusters = cluster_details['clusters']
            n_clusters = cluster_details['n_clusters']

            filled_clusters[label_file] = {'n_clusters': n_clusters, 'clusters': {}}
            for cluster_id, cluster in clusters.items():
                max_confidence_existing = -1
                max_confidence_bbox = ''
                for h_type_existing in cluster:
                    if cluster[h_type_existing]['score'] > max_confidence_existing:
                        max_confidence_existing = cluster[h_type_existing]['score']
                        max_confidence_bbox = cluster[h_type_existing]['bounding_box']

                filled_clusters[label_file]['clusters'][cluster_id] = {}
                for h_type in h_types:
                    if h_type in cluster:
                        filled_clusters[label_file]['clusters'][cluster_id][h_type] = cluster[h_type]
                    else:        
                        filled_clusters[label_file]['clusters'][cluster_id][h_type] = {'score': -1, 'bounding_box': ''}
                        for score, bbox in conformal_scores[label_file][h_type]:
                            if score == -1: continue
                            iou = calculate_iou(bbox, max_confidence_bbox)
                            if iou > IOU_THRESHOLD:
                                if score > filled_clusters[label_file]['clusters'][cluster_id][h_type]['score']:
                                    filled_clusters[label_file]['clusters'][cluster_id][h_type]['score'] = score
                                    filled_clusters[label_file]['clusters'][cluster_id][h_type]['bounding_box'] = bbox
        print('Completed Step 3: Filling clusters with predictions')

        calibration_confidence_scores = pd.read_csv('/research/projects/Cooper/CQ500_UQ/paper_prep/calibration_scores.csv')
        calibration_confidence_scores.head()

        calibration_scores_dict = {
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

        for _, row in tqdm(list(calibration_confidence_scores.iterrows())):
            for h_type in h_types: 
                for presence in ['Presence', 'Absence']:
                    calibration_scores_dict[h_type][presence.lower()].append(row[f'{h_type} {presence}'])
        for h_type in h_types:
            for presence in ['presence', 'absence']:
                calibration_scores_dict[h_type][presence] = sorted(calibration_scores_dict[h_type][presence])

        def find_insert_index(arr, ele):
            for i in range(len(arr)):
                if ele < arr[i]: # use < if we want to place the test sample below all equal calibration samples
                    return i
            return len(arr)

        filled_clusters_with_p_values = {}
        for label_file, cluster_details in tqdm(filled_clusters.items()):
            clusters = cluster_details['clusters']
            n_clusters = cluster_details['n_clusters']

            filled_clusters_with_p_values[label_file] = {'n_clusters': n_clusters, 'clusters': {}}
            for cluster_id, cluster in clusters.items():
                filled_clusters_with_p_values[label_file]['clusters'][cluster_id] = {}
                for h_type in h_types:
                    if cluster[h_type]['score'] == -1:
                        filled_clusters_with_p_values[label_file]['clusters'][cluster_id][h_type] = {'score': -1, 'bounding_box': '', 'absence_p_value': -1, 'presence_p_value': -1}
                    else:
                        score = cluster[h_type]['score']
                        presence_p_value = find_insert_index(calibration_scores_dict[h_type]['presence'], score) / len(calibration_scores_dict[h_type]['presence'])
                        absence_p_value = find_insert_index(calibration_scores_dict[h_type]['absence'], score) / len(calibration_scores_dict[h_type]['absence'])
                        filled_clusters_with_p_values[label_file]['clusters'][cluster_id][h_type] = {'score': score, 'bounding_box': cluster[h_type]['bounding_box'], 'absence_p_value': absence_p_value, 'presence_p_value': presence_p_value}
        print('Completed Step 4: Calculating p-values for each cluster')

        test_prediction_sets = {}
        for label_file, cluster_details in tqdm(filled_clusters_with_p_values.items()):
            clusters = cluster_details['clusters']
            n_clusters = cluster_details['n_clusters']

            test_prediction_sets[label_file] = {'n_clusters': n_clusters, 'clusters': {}, 'prediction_set': set()}
            for cluster_id, cluster in clusters.items():
                test_prediction_sets[label_file]['clusters'][cluster_id] = {}
                for h_type in h_types:
                    if cluster[h_type]['score'] == -1:
                        test_prediction_sets[label_file]['clusters'][cluster_id][h_type] = {'score': -1, 'bounding_box': '', 'absence_p_value': -1, 'presence_p_value': -1}
                    else:
                        score = cluster[h_type]['score']
                        presence_p_value = cluster[h_type]['presence_p_value']
                        absence_p_value = cluster[h_type]['absence_p_value']
                        
                        if presence_p_value > P_VALUE_THRESHOLD:
                            x, y, w, h = [float(x)*512 for x in cluster[h_type]['bounding_box'].split(' ')]
                            test_prediction_sets[label_file]['prediction_set'].add(f'{h_type} (C={score:.2f}) centered at ({x:.2f}, {y:.2f}) with width {w:.2f} and height {h:.2f}')
                        if absence_p_value > P_VALUE_THRESHOLD:
                            test_prediction_sets[label_file]['prediction_set'].add(f'No {h_type} (C={1-score:.2f}) in cluster {cluster_id}')
                        
                        test_prediction_sets[label_file]['clusters'][cluster_id][h_type] = {'score': score, 'bounding_box': cluster[h_type]['bounding_box'], 'absence_p_value': absence_p_value, 'presence_p_value': presence_p_value}

        print('Completed Step 5: Generating prediction sets')
        pickle.dump(test_prediction_sets, open(f'/research/projects/Cooper/CQ500_UQ/paper_prep/mcp_results/test_prediction_sets_p_{P_VALUE_THRESHOLD:.2f}_iou_{IOU_THRESHOLD:.2f}.pkl', 'wb'))
        print(f'Saved prediction sets to mcp_results/test_prediction_sets_p_{P_VALUE_THRESHOLD:.2f}_iou_{IOU_THRESHOLD:.2f}.pkl')
