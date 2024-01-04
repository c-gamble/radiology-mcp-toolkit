import os
from typing import List
from tqdm.auto import tqdm
from collections import defaultdict

int_to_label: dict = {0: 'IPH', 1: 'IVH', 2: 'SDH', 3: 'EDH', 4: 'SAH'}
h_types: List[str] = list(int_to_label.values())

# Read the text files you stored during your model's inference on your test dataset.
test_HNUs: dict = defaultdict(dict)
"""
test_HNUs = {
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
for prediction_file in tqdm(os.listdir("./Step 1: Raw Predictions/Test Inference Outputs/labels")):
    with open(os.path.join("./Step 1: Raw Predictions/Test Inference Outputs/labels", prediction_file), 'r') as f:
        lines: List[str] = f.readlines()

        classwise_predictions: dict = {h_type: [] for h_type in h_types}
        for line in lines:
            HNU: float = float(line.split(" ")[-1].split('\n')[0])
            label_int: int = int(float(line.split(" ")[0]))
            bounding_box: str = " ".join(line.split(" ")[1:5])
            classwise_predictions[int_to_label[label_int]].append([HNU, bounding_box])
    
    test_HNUs[prediction_file.split('.txt')[0]]: dict = classwise_predictions

# Save the formatted predictions (for tutorial purposes only).
with open("./Step 1: Raw Predictions/test_HNUs.txt", 'w') as f:
    f.write("This .txt file is for tutorial purposes only. Please avoid saving your dictionary output by combining multiple steps into a single .py file. If you must save your dictionary output, please use a .pkl file or other efficient storage method.\n\n")
    # print a formatted dictionary
    f.write("test_HNUs = {\n")
    for instance in test_HNUs:
        f.write(f"\t'{instance}': {{\n")
        for h_type in test_HNUs[instance]:
            f.write(f"\t\t'{h_type}': [\n")
            for prediction in test_HNUs[instance][h_type]:
                f.write(f"\t\t\t{prediction},\n")
            f.write("\t\t],\n")
        f.write("\t},\n")
    f.write("}\n")
