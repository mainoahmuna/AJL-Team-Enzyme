import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, class_names, fitzpatrick_scales_test):
    """
    Evaluate a trained model on the test set, analyzing performance across different Fitzpatrick Scale groups.
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {model.__class__.__name__} Model Performance")
    print(f"{'='*50}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute overall evaluation metrics
    overall_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    # Print overall results with better formatting
    print(f"\n{'='*20} Overall Performance {'='*20}")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"F1 Score: {overall_metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(overall_metrics['conf_matrix'])
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print(f"{'='*50}\n")

    # Evaluate performance across Fitzpatrick Scale categories
    fitzpatrick_metrics = {}
    unique_scales = np.unique(fitzpatrick_scales_test)

    for scale in unique_scales:
        indices = np.where(fitzpatrick_scales_test == scale)[0]
        if len(indices) == 0:
            continue

        y_test_subset = y_test[indices]
        y_pred_subset = y_pred[indices]

        scale_metrics = {
            "accuracy": accuracy_score(y_test_subset, y_pred_subset),
            "precision": precision_score(y_test_subset, y_pred_subset, average="weighted", zero_division=0),
            "recall": recall_score(y_test_subset, y_pred_subset, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test_subset, y_pred_subset, average="weighted", zero_division=0),
            "conf_matrix": confusion_matrix(y_test_subset, y_pred_subset)
        }

        fitzpatrick_metrics[scale] = scale_metrics

        # Print performance for this Fitzpatrick Scale with better formatting
        print(f"\n{'-'*25} Performance for Fitzpatrick Scale {scale} {'-'*25}")
        print(f"Accuracy: {scale_metrics['accuracy']:.4f}")
        print(f"Precision: {scale_metrics['precision']:.4f}")
        print(f"Recall: {scale_metrics['recall']:.4f}")
        print(f"F1 Score: {scale_metrics['f1_score']:.4f}")
        print("Confusion Matrix:")
        print(scale_metrics["conf_matrix"])
        print(f"{'-'*50}\n")

    return {
        "overall": overall_metrics,
        "fitzpatrick_metrics": fitzpatrick_metrics
    }


def evaluate_multiple_models(models, X_test, y_test, class_names, fitzpatrick_scales_test):
    """
    Evaluate multiple trained models and return their metrics, analyzing performance across different Fitzpatrick Scale groups.

    Parameters:
        models (list): List of trained models.
        X_test (ndarray): Test data.
        y_test (ndarray): True test labels.
        class_names (list): List of class labels.
        fitzpatrick_scales_test (ndarray): Fitzpatrick scale values corresponding to X_test.

    Returns:
        list: A list of dictionaries containing evaluation metrics for each model.
    """
    results = []

    for model in models:
        result = evaluate_model(model, X_test, y_test, class_names, fitzpatrick_scales_test)
        results.append(result)

    # Sort results by overall accuracy
    results = sorted(results, key=lambda x: x["overall"]["accuracy"], reverse=True)

    print("\nModel Evaluation Results:")
    for res in results:
        print(f"{res['overall']['model']}: Accuracy={res['overall']['accuracy']:.4f}, F1={res['overall']['f1_score']:.4f}")

    return results