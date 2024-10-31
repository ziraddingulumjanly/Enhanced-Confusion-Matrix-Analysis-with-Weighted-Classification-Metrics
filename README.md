### Project Overview

In classification tasks, confusion matrices are essential for evaluating model performance by showing the count of correct and incorrect predictions for each class. However, traditional confusion matrices treat each prediction outcome as binary (correct or incorrect), disregarding the confidence or strength associated with each prediction. This project aims to address this limitation by introducing **weighted feature values**—confidence measures associated with each classification—into the confusion matrix analysis. 

### Original Problem

The limitation of traditional confusion matrices lies in their binary nature, which fails to capture the varying degrees of prediction confidence that often exist in real-world data. This project develops a weighted confusion matrix approach that factors in the prediction strength, providing a more detailed evaluation of model performance. By incorporating weighted values, we can obtain more meaningful accuracy, recall, and precision metrics that reflect the classifier's behavior beyond mere correctness.

### Original Confusion Matrix

The original (unweighted) confusion matrix provides a baseline for comparison:

|            | Predicted A | Predicted B | Predicted C |
|------------|-------------|-------------|-------------|
| **True A** | 5           | 2           | 3           |
| **True B** | 2           | 8           | 0           |
| **True C** | 2           | 0           | 8           |

### What the Code Does

This code first constructs a **weighted confusion matrix** by summing weighted feature values for each predicted class, thereby capturing classification confidence. It then calculates essential performance metrics—**accuracy**, **recall**, and **precision**—based on this weighted matrix. Additionally, it normalizes the weighted matrix rows so that each row sums to 1, providing an interpretable distribution for each true class. Finally, the code compares the normalized matrix with the original confusion matrix, generating an **error percentage matrix** to assess the divergence between weighted and binary outcomes. 

This enhanced approach offers a comprehensive view of classifier performance, making it particularly useful in cases where prediction confidence varies and influences the assessment of model accuracy.
