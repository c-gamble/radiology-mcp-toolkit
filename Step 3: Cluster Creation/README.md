# Step 3: Cluster Creation

## Prerequisites
- [ ] A predefined IoU threshold value (edit line 11 in ```create_clusters.py``` if you want to use a value other than 0.25)
    - It is recommended that you optimize this value using your desired metric for your specific task on the test dataset by varying the IoU threshold between 0.0 and 1.0 at intervals of 0.05 (or whichever granularity your compute access can support).
- [ ] Suppressed test inference outputs formatted as shown in ```Step 2: Class-wise NMS/test_predictions_retained.txt```

## Return Values
- [x] Dictionary containing cluster counts and cluster details for each sample. A cluster is defined by any bounding box whose IoU with all other bounding boxes was less than the given threshold.
    - For the purpose of this tutorial, we have split each step into a separate file, so we must save our dictionary. However, in practice, it is more storage-efficient to combine as many steps as possible into a single ```.py``` file. If you do choose to save the dictionary output of this step, we recommended that you use a ```.pkl``` file rather than a ```.txt``` file to reduce read/write time.
