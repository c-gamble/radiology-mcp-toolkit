# Step 1: Raw Predictions

## Prerequisite
- [ ] A trained model and accompanying weights
- [ ] A directory of test images organized as shown in ```example_data/```
## Return Values
- [x] Dictionary containing results from your model's inference on the test dataset
    - For the purpose of this tutorial, we have split each step into a separate file, so we must save our dictionary. However, in practice, it is more storage-efficient to combine as many steps as possible into a single ```.py``` file. If you do choose to save the dictionary output of this step, we recommended that you use a ```.pkl``` file rather than a ```.txt``` file to reduce read/write time.
