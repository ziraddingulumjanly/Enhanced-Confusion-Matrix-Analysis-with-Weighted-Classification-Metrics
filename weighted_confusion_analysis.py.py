import pandas as pd

class_a_data = {
    'Sample': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sum Weight(A)': [0.324, 0.242, 0.084, 0.745, 0.762, 0.569, 0.546, 0.611, 0.055, 0.026],
    'Sum Weight(B)': [0.189, 0.129, 0.861, 0.184, 0.193, 0.357, 0.434, 0.375, 0.029, 0.966],
    'Sum Weight(C)': [0.487, 0.629, 0.055, 0.071, 0.045, 0.074, 0.020, 0.014, 0.916, 0.008],
    'Classification': ['C', 'C', 'B', 'A', 'A', 'A', 'A', 'A', 'C', 'B'],
    'Result': ['INCORRECT', 'INCORRECT', 'INCORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'INCORRECT', 'INCORRECT']
}

class_b_data = {
    'Sample': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sum Weight(A)': [0.200, 0.561, 0.107, 0.090, 0.280, 0.224, 0.428, 0.076, 0.748, 0.147],
    'Sum Weight(B)': [0.694, 0.148, 0.866, 0.839, 0.700, 0.761, 0.566, 0.906, 0.215, 0.783],
    'Sum Weight(C)': [0.106, 0.291, 0.027, 0.071, 0.020, 0.015, 0.006, 0.018, 0.037, 0.070],
    'Classification': ['B', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'A', 'B'],
    'Result': ['CORRECT', 'INCORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'INCORRECT', 'CORRECT']
}

class_c_data = {
    'Sample': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sum Weight(A)': [0.049, 0.007, 0.130, 0.016, 0.101, 0.098, 0.263, 0.646, 0.699, 0.082],
    'Sum Weight(B)': [0.040, 0.003, 0.031, 0.005, 0.078, 0.122, 0.128, 0.149, 0.075, 0.052],
    'Sum Weight(C)': [0.911, 0.990, 0.839, 0.979, 0.821, 0.780, 0.609, 0.205, 0.226, 0.866],
    'Classification': ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'A', 'A', 'C'],
    'Result': ['CORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'CORRECT', 'INCORRECT', 'INCORRECT', 'CORRECT']
}

# Creating DataFrames from the data
df_class_a = pd.DataFrame(class_a_data)
df_class_b = pd.DataFrame(class_b_data)
df_class_c = pd.DataFrame(class_c_data)

# Computing the Weighted Confusion Matrix
weighted_conf_matrix = pd.DataFrame({
    'A': [
        df_class_a[df_class_a['Classification'] == 'A']['Sum Weight(A)'].sum(),
        df_class_b[df_class_b['Classification'] == 'A']['Sum Weight(A)'].sum(),
        df_class_c[df_class_c['Classification'] == 'A']['Sum Weight(A)'].sum()
    ],
    'B': [
        df_class_a[df_class_a['Classification'] == 'B']['Sum Weight(B)'].sum(),
        df_class_b[df_class_b['Classification'] == 'B']['Sum Weight(B)'].sum(),
        df_class_c[df_class_c['Classification'] == 'B']['Sum Weight(B)'].sum()
    ],
    'C': [
        df_class_a[df_class_a['Classification'] == 'C']['Sum Weight(C)'].sum(),
        df_class_b[df_class_b['Classification'] == 'C']['Sum Weight(C)'].sum(),
        df_class_c[df_class_c['Classification'] == 'C']['Sum Weight(C)'].sum()
    ]
}, index=['True A', 'True B', 'True C'])

print("Weighted Confusion Matrix")
print(weighted_conf_matrix)

# Calculating Accuracy, Recall, and Precision for the Weighted Confusion Matrix

true_positives = {
    'A': weighted_conf_matrix.loc['True A', 'A'],
    'B': weighted_conf_matrix.loc['True B', 'B'],
    'C': weighted_conf_matrix.loc['True C', 'C']
}

actual_totals = {
    'A': weighted_conf_matrix.loc['True A'].sum(),
    'B': weighted_conf_matrix.loc['True B'].sum(),
    'C': weighted_conf_matrix.loc['True C'].sum()
}

predicted_totals = {
    'A': weighted_conf_matrix['A'].sum(),
    'B': weighted_conf_matrix['B'].sum(),
    'C': weighted_conf_matrix['C'].sum()
}

# Calculating Recall and Precision
recall = {cls: true_positives[cls] / actual_totals[cls] for cls in true_positives}
precision = {cls: true_positives[cls] / predicted_totals[cls] for cls in true_positives}

# Calculating overall accuracy
total_true_positives = sum(true_positives.values())
total_weights = weighted_conf_matrix.values.sum()
accuracy = total_true_positives / total_weights

print("Classification Metrics")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("Recall for each class:")
for cls, rec in recall.items():
    print(f" - Class {cls}: {rec * 100:.2f}%")
print("Precision for each class:")
for cls, prec in precision.items():
    print(f" - Class {cls}: {prec * 100:.2f}%")
# ------------------------------------------------------------------------------------

# Normalizing the Confusion Matrix
normalized_conf_matrix = weighted_conf_matrix.div(weighted_conf_matrix.sum(axis=1), axis=0)
print(" Normalized Confusion Matrix")
print(normalized_conf_matrix)

# Calculating Error Percentage Matrix Compared to the Original Matrix
original_conf_matrix = pd.DataFrame({
    'A': [5, 2, 3],
    'B': [2, 8, 0],
    'C': [2, 0, 8]
}, index=['True A', 'True B', 'True C'])

# ------------------------------------------------------------------------------------
# Calculating the error percentage matrix
error_percent_matrix = abs(normalized_conf_matrix - original_conf_matrix) / original_conf_matrix * 100
error_percent_matrix = error_percent_matrix.fillna(0)

print("Error Percentage Matrix (compared to original matrix)")
print(error_percent_matrix)
