# Step 2: Class-wise NMS

## Prerequisites
- [ ] Test inference outputs formatted as shown in ```Step 1: Raw Predictions/Test Inference Outputs```

## Return Values
- [x] Dictionary containing one bounding box and corresponding HNU for each hemorrhage class for each sample
    - For the purpose of this tutorial, we have split each step into a separate file, so we must save our dictionary. However, in practice, it is more storage-efficient to combine as many steps as possible into a single ```.py``` file. If you do choose to save the dictionary output of this step, we recommended that you use a ```.pkl``` file rather than a ```.txt``` file to reduce read/write time.
