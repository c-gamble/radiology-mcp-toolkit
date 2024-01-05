# Step 6: Prediction Sets

## Prerequisites
- [ ] A predefined conformal score threshold value (edit line 10 in ```generate_predictions.py``` if you want to use a value other than 0.6)
    - It is recommended that you optimize this value using your desired metric for your specific task on the test dataset by varying the conformal score threshold between 0.0 and 1.0 at intervals of 0.05 (or whichever granularity your compute access can support).
- [ ] Dictionary of filled clusters formatted as shown in ```Step 4 and Step 5: Cluster Filling/test_clusters_filled.txt```
- [ ] HNU values from your calibration dataset formatted as shown in ```Step 0: Calibration/calibration_HNUs.csv```

## Return Values
- [x] Dictionary containing final prediction sets (as well as cluster counts and filled cluster details) for each sample. Prediction sets will include strings specifying the hemorrhage type, its presence or absence, its HNU value, and a bounding box if it is predicted as present.
    - For the purpose of this tutorial, we have split each step into a separate file, so we must save our dictionary. However, in practice, it is more storage-efficient to combine as many steps as possible into a single ```.py``` file. If you do choose to save the dictionary output of this step, we recommended that you use a ```.pkl``` file rather than a ```.txt``` file to reduce read/write time.
