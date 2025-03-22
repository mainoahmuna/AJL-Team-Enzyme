# AI for Inclusive Dermatology - Kaggle Competition

## Overview
AI is transforming healthcare, yet dermatology AI tools often underperform for people with darker skin tones due to a lack of diverse training data. This can lead to diagnostic errors, delayed treatments, and health disparities for underserved communities.

This competition, organized by Break Through Tech and the Algorithmic Justice League, aims to build an inclusive machine learning model for dermatology that accurately classifies 21 different skin conditions across diverse skin tones. The competition evaluates submissions based on a weighted average F1 score and also encourages participants to focus on fairness and explainability in their models.

## Goals
1. **Develop a robust classification model**
   - Train a model to classify 21 different skin conditions using provided datasets.
   - Improve model performance using machine learning techniques such as data augmentation and transfer learning.

2. **Ensure fairness and explainability**
   - Consider how AI models impact historically marginalized communities in healthcare.
   - Use fairness and explainability tools to analyze model biases and illustrate decision-making.
   - Provide visualizations, reports, and creative storytelling techniques to highlight how marginalized groups were centered in model development.

## Our Approach
### Baseline Model: Logistic Regression
- Implemented logistic regression as a simple baseline model to establish a starting point for performance comparison.

### Convolutional Neural Network (CNN)
- Developed a CNN to capture spatial features in skin condition images.
- Applied data augmentation techniques to improve generalization and ensure balanced representation across skin tones.

### Ensemble Methods: Bagging with CNN and Vision Transformer (ViT)
- Combined predictions from CNN and Vision Transformer (ViT) using bagging to enhance model robustness and accuracy.
- Leveraged transfer learning with pre-trained ViT to improve feature extraction and classification.

## Evaluation Metric
The primary evaluation metric is the weighted **F1 Score**, calculated as:

\[ F1 = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}} \]

This metric ensures a balance between precision and recall while accounting for class imbalances.

## Resources
- [Kaggle Competition Page](#)
- [Fairlearn Documentation](https://fairlearn.org/)
- [Algorithmic Justice League](https://www.ajlunited.org/)

Join us in creating equitable AI solutions in dermatology!

