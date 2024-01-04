# Step 0: Calibration
You should begin this step by running your model on your calibration dataset and saving the results according to the formatting described below. If you are using a YOLO model, you should set the following arguments to avoid default filtering operaitons:  ```conf=0 iou=1 max_det=10000```. If you are using another model, set the necessary parameters to acquire as many predictions per sample as possible.

## Prerequisites
- [ ] A trained model and accompanying weights
- [ ] A directory of ```.txt``` files named according to their instance identifier (i.e., ```SOPInstanceUID``` for DICOM-encoded datasets, or simply the patient, scan, and slice number for other encodings). 
    - Each file should contain line-separated predictions for your calibration dataset.
    - Each line should contain a space-separated label and bounding box. 
    - The label should be a single integer between 0 and the number of classes in your task minus 1, inclusive. 
    - The bounding box should be in normalized ```xywh``` form. For YOLO models, this is accessible via the ```results.boxes.xywhn``` field. For other use cases, the ```(x, y)``` ordered pair corresponds to the normalized center of the predicted bounding box. ```w``` is the width and ```h``` is the height, also both normalized to the image size. 
    - Please see ```example_predictions/``` for examples of this formatting. Note that although each example only contains 10 predictions, it is encouraged that you acquire as many predictions per sample as possible for this and all subsequent inference steps. This will ensure that all filtering happens according to MCP, rather than according to predefined selection criteria in either the YOLO pipeline or another out-of-the-box deep learning tool. 
## Return Values
- [x] ```.csv``` file with the following headers:
    - ```Sample```: the name of a given sample.
    - ```{Class} Presence```: For each class, the calibration score for its presence.
    - ```{Class} Absence```: For each class, the calibration score for its absence.
