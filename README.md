# Equitable Dermatology AI: Skin Condition Classification
---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Atmaja Patil | @atmpats | Built CNN Model, Data Preprocessing |
| Khushi Chandra | @chandrakhushi | EDA & Data preprocessing |
| Dureti Shemsi | @DuretiShemsi | Built, Trained & Tested CNN Model, Data augmentation |
| Victoria Worthington | @VicaWorth | EDA & Feature Engineering, Training & Testing CNN Model |
| Mainoah Muna | @mainoahmuna | Setup Github & Evaluation Metrics, Trained Baseline Model (Logistic Regression), Trained Ensemble Methods Model |

---

## **🎯 Project Highlights**

**Example:**

* Built a \[insert model type\] using \[techniques used\] to solve \[Kaggle competition task\]
* Achieved an F1 score of \[insert score\] and a ranking of \[insert ranking out of participating teams\] on the final Kaggle Leaderboard
* Used \[explainability tool\] to interpret model decisions
* Implemented \[data preprocessing method\] to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)
🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## **🏗️ Project Overview**

This project, developed as part of the Break Through Tech AI Program's Spring 2025 AI Studio in collaboration with the Algorithmic Justice League (AJL), addresses the challenge of building a fair and accurate machine learning model for classifying skin conditions from images. The focus is on ensuring equitable performance across diverse skin tones, combating the common issue of bias in dermatology AI where models trained predominantly on lighter skin tones perform poorly on darker skin tones.

**Objective:** The primary goal is to develop a model that accurately classifies 21 different skin conditions using image data provided in the Kaggle competition. The model must perform consistently well across a range of Fitzpatrick skin types, mitigating biases and promoting equitable healthcare outcomes.

**Real-world Significance:** AI's increasing role in healthcare, particularly in dermatology, necessitates models that are free from bias. Biased datasets can lead to misdiagnosis and delayed or inappropriate treatment for individuals with darker skin, exacerbating existing health disparities. This project aims to contribute to more equitable healthcare by creating an AI model that is inclusive and reliable for all patients, regardless of their skin tone. Accurate and equitable AI-powered diagnostic tools can lead to earlier and more effective treatment, ultimately improving patient outcomes.

---

## **📊 Data Exploration**

The primary dataset for this competition is provided by Kaggle and consists of images of various skin conditions along with associated metadata.

**Dataset Description:** The dataset includes:

*   **Images:** JPG images of skin conditions.
*   **Metadata:**
    *   `md5hash`: A unique identifier for each image.
    *   `fitzpatrick_scale`: A numerical scale (1-6) representing skin tone, with 1 being the lightest and 6 the darkest.  This is a key feature for evaluating fairness.
    *   `fitzpatrick_centaur`: Another representation of the Fitzpatrick scale.
    *   `label`: The target variable, indicating the specific skin condition (one of 21 classes).
    *   `nine_partition_label`: A broader categorization of the skin condition into nine categories.
    *   `three_partition_label`: An even broader categorization into three categories.
    *   `qc`: Quality control flags (mostly missing, as noted in our analysis).
    *   `ddi_scale`: Another scale, purpose not fully documented in the provided materials.
    *   `file_path`: The path to the image file within the dataset.

**Class Imbalance:**

![image](https://github.com/user-attachments/assets/5dfce175-e414-4579-bca0-f9b0ac8f025a)

---

## **🧠 Model Development**

**Logistic Regression:**

* Implemented logistic regression as a starting point for performance benchmarking.

**CNN Model:**
* We implemented a custom Convolutional Neural Network (CNN) model for this classification task. We did *not* use transfer learning in this version, opting for a model trained from scratch to better understand the impact of our data preprocessing and fairness considerations.

The CNN architecture consists of five convolutional blocks followed by global average pooling and dense layers:

*   **Convolutional Blocks:** Each block contains:
    *   Convolutional layers (`Conv2D`) with ReLU activation and 'same' padding.
    *   Batch Normalization (`BatchNormalization`).
    *   Max Pooling (`MaxPooling2D`).
    *   Dropout (`Dropout`) for regularization.
    * The number of filters increases in each subsequent block (32, 64, 128, 256, 512).
* **Global Average Pooling**:
    * MaxPooling
*   **Dense Layers:**
    *   A dense layer with 256 units and ReLU activation, with L2 regularization.
    *   Batch Normalization
    *   Dropout (rate 0.5).
    *   A final dense layer with a number of units equal to the number of classes (21) and softmax activation for multi-class classification.

**Training Strategy:**

* Used 80/20 train-validation split.
* Evaluated performance using the weighted F1 score.
---

## **📈 Results & Key Findings**

<> (**Training Results (Summary):**)

## Evaluation
We used several metrics to evaluate our model:
* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## **🖼️ Impact Narrative**

Ensuring fairness and explainability is a core requirement of this project. We plan to conduct the following analyses:

1.  **Stratified Evaluation:** We will evaluate the model's performance separately for each Fitzpatrick skin tone group. This will allow us to identify any significant disparities in accuracy or other metrics across different skin tones. We will report these stratified metrics in detail.

2.  **Confusion Matrices per Fitzpatrick Scale:** We will generate confusion matrices for each Fitzpatrick skin tone group. This will visualize which classes are most frequently confused within each group, providing insights into potential biases related to specific conditions and skin tones.

3.  **Explainability with Grad-CAM:** We will implement Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions of the input images that most strongly influence the model's predictions. We will analyze these visualizations across different skin tones and conditions to identify potential biases in the model's decision-making process. For instance, we will investigate whether the model focuses on irrelevant features or exhibits different patterns of attention for different skin tones.
   
4. **Bias Mitigation Techniques:** If significant biases are identified, we will explore mitigation techniques, such as:
    *   **Data Re-sampling:** Oversampling under-represented skin tones or conditions.
    *   **Algorithmic Adjustments:** Exploring fairness-aware loss functions or regularization techniques.

By systematically addressing fairness and explainability, we aim to develop a dermatology AI model that is both accurate and equitable for all individuals. We will update this section with our findings and mitigation strategies as we progress.

---

## **🚀 Next Steps & Future Improvements**


*   **Hyperparameter Optimization:** Conduct a more systematic search for optimal hyperparameters (learning rate, dropout rate, data augmentation parameters, optimizer) using techniques like grid search or Bayesian optimization.
*   **Alternative Pre-trained Models:** Experiment with other pre-trained models (e.g., EfficientNet, ResNet) to compare performance.
*   **Advanced Data Augmentation:** Investigate more sophisticated data augmentation techniques, such as CutMix or MixUp.
*   **Ensemble Methods:** Combine predictions from multiple models (e.g., different architectures or training runs) to potentially improve robustness and accuracy.
*   **Further Error Analysis:** Carefully examine misclassified images to identify patterns and potential areas for improvement in the data or model.
* Logistic Regression Model Implementation
* Evaluation Function
* Data Preprocessing Functions

---

## **📄 References & Additional Resources**

* [Tensor Flow Transfer Learning Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
* [Tensor Flow Data Augmentation Tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation)
* [CNN Comprehensive Guide](https://medium.com/@navarai/unveiling-the-diversity-a-comprehensive-guide-to-types-of-cnn-architectures-9d70da0b4521)
* [Keras Regularization Documentation](https://keras.io/api/layers/regularization_layers/dropout/)

---
