import os
from tqdm.auto import tqdm

# Read the formatted HNUs from Step 1.
with open("./Step 1: Raw Predictions/test_HNUs.txt", 'r') as f:
    lines = f.read().split('\n\n')[1]
    exec(lines) # this will define dictionary of HNUs called test_HNUs

test_predictions_retained: dict = {}
"""
test_predictions_retained = {
    'instance identifier': {
        'IPH': {
            'HNU': HNU (-1 if no prediction),
            'bounding_box': bounding box in xywhn format ('' if no prediction),
        },
        'IVH': {
            ...
        }
    }
"""
for instance_identifier, prediction_dict in tqdm(list(test_HNUs.items())):
    
    test_predictions_retained[instance_identifier]: dict = {}

    for h_type, predictions in prediction_dict.items():

        max_HNU_value: float = -1
        max_HNU_value_bounding_box: str = ''
        
        for bounding_box in predictions:
            if bounding_box[0] > max_HNU_value:
                max_HNU_value = bounding_box[0]
                max_HNU_value_bounding_box = bounding_box[1]
        
        test_predictions_retained[instance_identifier][h_type]: dict = {
            'HNU': max_HNU_value,
            'bounding_box': max_HNU_value_bounding_box
        }

# Save the retained predictions.
with open("./Step 2: Class-wise NMS/test_predictions_suppressed.txt", 'w') as f:
    f.write("This .txt file is for tutorial purposes only. Please avoid saving your dictionary output by combining multiple steps into a single .py file. If you must save your dictionary output, please use a .pkl file or other efficient storage method.\n\n")
    # print a formatted dictionary
    f.write("test_predictions_retained = {\n")
    for instance in test_predictions_retained:
        f.write(f"    '{instance}': {{\n")
        for h_type in test_predictions_retained[instance]:
            f.write(f"        '{h_type}': {{\n")
            for key in test_predictions_retained[instance][h_type]:
                f.write(f"            '{key}': {test_predictions_retained[instance][h_type][key]},\n")
            f.write("        },\n")
        f.write("    },\n")