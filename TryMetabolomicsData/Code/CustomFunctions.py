#all necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#imports for ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#imports for random forest
from sklearn.ensemble import RandomForestClassifier

#creates roc curve
def create_ROC_curve(x_dimension, y_dimension, ml_models, false_positive_rates, true_positive_rates, roc_auc_scores, title):
    #set figure size
    plt.figure(figsize=(x_dimension, y_dimension))

    #add random forest
    for model, frp, tpr, rcs in zip(ml_models, false_positive_rates, true_positive_rates, roc_auc_scores):
        plt.plot(frp, tpr, label = model + ' (area = %0.2f)' % rcs)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def visualize_confusion_matrix(ml_model, actual_values, test_values, labels):
    predictions = ml_model.predict(test_values)
    confusion_matrix = metrics.confusion_matrix(actual_values, predictions)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
    display.plot()

def visualize_feature_importance(ml_model, x_dimension, y_dimension, title, color, variable_list):
    importances = ml_model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(x_dimension, y_dimension))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color = color, align='center')
    plt.yticks(range(len(indices)), [variable_list[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


def estimate_ml_metrics(ml_model, model_name, actual_values, test_values, runs):
    timings = []
    number_of_runs = runs
    for i in range(number_of_runs):
        #run model estimate time
        start = time.perf_counter()
        predictions = ml_model.predict(test_values)
        end = time.perf_counter()
        timings.append(end - start)

    #all estimates ± standard deviation if applicable
    latency = f"{np.mean(timings):.4f} ± {np.std(timings):.4f} seconds"
    accuracy = metrics.accuracy_score(actual_values, predictions)
    precision = metrics.precision_score(actual_values, predictions, average = 'weighted')
    recall = metrics.recall_score(actual_values, predictions, average = 'weighted')
    f1 = metrics.f1_score(actual_values, predictions, average = 'weighted')

    #organize as dictionary
    all_metrics = {'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1' : f1, 'Latency': latency}
    return all_metrics

def create_comparison_frame(all_metrics_list):
    comparisons = np.zeros((len(all_metrics_list), 5)) # 0 - name,  1 - accuracy, 2 - precision, 3 - recall, 4 - f1, 5 - latency

    #add all metrics from all models
    for metrics in all_metrics_list:
        comparisons[0, :].append(metrics['Model'])
        comparisons[1, :].append(metrics['Accuracy'])
        comparisons[2, :].append(metrics['Precision'])
        comparisons[3, :].append(metrics['Recall'])
        comparisons[4, :].append(metrics['F1'])
        comparisons[5, :].append(metrics['Latency'])

    #store metrics in comparison frame comparisons[0, :]
    comparison_frame = pd.DataFrame({'Model': comparisons[0, :], 'accuracy': comparisons[1, :], 'Precision': comparisons[2, :], 'Recall': comparisons[3, :], 'F1' : comparisons[4, :], 'Latency': comparisons[5, :]}) 
    return comparison_frame