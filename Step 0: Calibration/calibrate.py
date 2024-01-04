import os
import pandas as pd
from typing import List
from collections import defaultdict
from tqdm.auto import tqdm

int_to_label: dict = {0: 'IPH', 1: 'IVH', 2: 'SDH', 3: 'EDH', 4: 'SAH'}
h_types: List[str] = list(int_to_label.values())


# Begin by reading the text files you stored during your model's inference on your calibration dataset.
calibration_HNUs: dict = defaultdict(dict)
"""
calibration_HNUs = {
    'instance identifier': {
        'IPH': {
            [HNU, bounding box in xywhn format],
            [HNU, bounding box in xywhn format],
            ...
        },
        'IVH': ...
    }
}
"""
for prediction_file in tqdm(os.listdir("./Step 0: Calibration/Calibration Inference Outputs/labels")):
    with open(os.path.join("./Step 0: Calibration/Calibration Inference Outputs/labels", prediction_file), 'r') as f:
        lines: List[str] = f.readlines()

        classwise_predictions: dict = {h_type: [] for h_type in h_types}
        for line in lines:
            HNU: float = float(line.split(" ")[-1].split('\n')[0])
            label_int: int = int(float(line.split(" ")[0]))
            bounding_box: str = " ".join(line.split(" ")[1:5])
            classwise_predictions[int_to_label[label_int]].append([HNU, bounding_box])
    
    calibration_HNUs[prediction_file.split('.txt')[0]]: dict = classwise_predictions

# Next, perform suppression such that only the highest scoring bounding box for each class is kept.
calibration_predictions_retained: dict = {}
"""
calibration_predictions_retained = {
    'instance identifier': {
        'IPH': {
            'HNU': HNU (-1 if no prediction),
            'bounding_box': bounding box in xywhn format ('' if no prediction),
        },
        'IVH': {
            ...
        }
    }
}
"""
for instance_identifier, prediction_dict in tqdm(list(calibration_HNUs.items())):
    
    calibration_predictions_retained[instance_identifier]: dict = {}

    for h_type, predictions in prediction_dict.items():

        max_HNU_value: float = -1
        max_HNU_value_bounding_box: str = ''
        
        for bounding_box in predictions:
            if bounding_box[0] > max_HNU_value:
                max_HNU_value = bounding_box[0]
                max_HNU_value_bounding_box = bounding_box[1]
        
        calibration_predictions_retained[instance_identifier][h_type]: dict = {
            'HNU': max_HNU_value,
            'bounding_box': max_HNU_value_bounding_box
        }   

# Now, condense the retained predictions to only store the hemorrhage type and calibration HNU.
presence_calibration_HNUs: dict = {}
"""
presence_calibration_HNUs = {
    'instance identifier': {
        'IPH': HNU,
        'IVH': HNU,
        ...
    }
}
"""
for instance_identifier, prediction_dict in tqdm(list(calibration_predictions_retained.items())):
    
    presence_calibration_HNUs[instance_identifier]: dict = {}

    for h_type, prediction in prediction_dict.items():
        if prediction['bounding_box'] == '':
            presence_calibration_HNUs[instance_identifier][h_type] = 0
            continue
        else: 
            presence_calibration_HNUs[instance_identifier][h_type] = prediction['HNU']

# Finally, create and save the .csv file containing the calibration HNUs.
final_calibration_HNUs: dict = {'Sample': [], f'{h_types[0]} Presence': [], f'{h_types[1]} Presence': [], f'{h_types[2]} Presence': [], f'{h_types[3]} Presence': [], f'{h_types[4]} Presence': [], f'{h_types[0]} Absence': [], f'{h_types[1]} Absence': [], f'{h_types[2]} Absence': [], f'{h_types[3]} Absence': [], f'{h_types[4]} Absence': []}
for instance_identifier, prediction_dict in tqdm(list(presence_calibration_HNUs.items())):
    final_calibration_HNUs['Sample'].append(instance_identifier)
    for h_type in h_types:
        final_calibration_HNUs[f'{h_type} Presence'].append(prediction_dict[h_type])
        final_calibration_HNUs[f'{h_type} Absence'].append(1 - prediction_dict[h_type])
final_calibration_HNUs: pd.DataFrame = pd.DataFrame(final_calibration_HNUs)
final_calibration_HNUs.to_csv('./Step 0: Calibration/calibration_HNUs.csv', index=False)
