 # Step 1: Raw Predictions


## Prerequisites
- [ ] A trained model and accompanying weights
- [ ] A directory of ```.txt``` files named according to their instance identifier (i.e., ```SOPInstanceUID``` for DICOM-encoded datasets, or simply the patient, scan, and slice number for other encodings). 
    - Each file should contain line-separated predictions for your calibration dataset.
    - Each line should contain a space-separated label and bounding box. 
    - The label should be a single integer between 0 and the number of classes in your task minus 1, inclusive. 
    - The bounding box should be in normalized ```xywh``` form. For YOLO models, this is accessible via the ```results.boxes.xywhn``` field. For other use cases, the ```(x, y)``` ordered pair corresponds to the normalized center of the predicted bounding box. ```w``` is the width and ```h``` is the height, also both normalized to the image size. 
    - Please see ```example_predictions/``` for examples of this formatting. Note that although each example only contains 10 predictions, it is encouraged that you acquire as many predictions per sample as possible for this and all subsequent inference steps. This will ensure that all filtering happens according to MCP, rather than according to predefined selection criteria in either the YOLO pipeline or another out-of-the-box deep learning tool. 
## Return Values
- [x] Dictionary containing results from your model's inference on the test dataset
    - For the purpose of this tutorial, we have split each step into a separate file, so we must save our dictionary. However, in practice, it is more storage-efficient to combine as many steps as possible into a single ```.py``` file. If you do choose to save the dictionary output of this step, we recommended that you use a ```.pkl``` file rather than a ```.txt``` file to reduce read/write time.
