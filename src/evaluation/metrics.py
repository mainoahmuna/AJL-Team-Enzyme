import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate a trained model on the test set.

    Parameters:
        model (sklearn model): The trained machine learning model.
        X_test (ndarray): Test data.
        y_test (ndarray): True test labels.
        class_names (list): List of class labels.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    print(f"\nEvaluating {model.__class__.__name__}...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display results
    print(f"\n{model.__class__.__name__} Performance:")
    # Precision accounts for the number of true positive results divided by the sum of true positive and false positive results
    print(f"Accuracy: {accuracy:.4f}")
    # Recall (sensitivity) measures the ability of a model to find all the relevant cases (true positive rate)
    print(f"Precision: {precision:.4f}")
    # F1 score is the harmonic mean of precision and recall
    print(f"Recall: {recall:.4f}")
    # Confusion matrix to show the number of correct and incorrect predictions
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return {
        "model": model.__class__.__name__,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "conf_matrix": conf_matrix,
    # Return the computed metrics in a dictionary
    }


def evaluate_multiple_models(models, X_test, y_test, class_names):
    """
    Evaluate multiple trained models and return their metrics.

    Parameters:
        models (list): List of trained models.
        X_test (ndarray): Test data.
        y_test (ndarray): True test labels.
        class_names (list): List of class labels.

    Returns:
        list: A list of dictionaries containing evaluation metrics for each model.
    """
    results = []

    for model in models:
        result = evaluate_model(model, X_test, y_test, class_names)
        results.append(result)

    # Sort results by accuracy
    results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    print("\nModel Evaluation Results:")
    for res in results:
        print(f"{res['model']}: Accuracy={res['accuracy']:.4f}, F1={res['f1_score']:.4f}")

    return results