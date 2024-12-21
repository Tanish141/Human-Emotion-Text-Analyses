# 📘 Emotion Classifier Using Logistic Regression

This project implements an Emotion Classifier that predicts the emotion of a given text (e.g., Angry, Sad, Happy, Neutral) using Logistic Regression. It includes features like dataset preprocessing, model training, evaluation, and a feedback loop for continuous learning.

---

## 🚀 Features

- Dynamic Dataset Loading: Automatically loads datasets with enhanced error handling.

- Text Preprocessing: Uses TF-IDF Vectorization to transform text data.

- Logistic Regression: Implements a robust model with dynamic class weighting.

- Feedback Loop: Allows users to correct predictions and retrain the model in real-time.

- Emotion Mapping: Maps emojis or text-based emotions to predefined categories.

- Interactive Inference: Provides real-time emotion prediction and feedback handling.

---

## 🛠️ Technologies Used

- Python 🐍

- Pandas: For data manipulation and loading.

- NumPy: For numerical operations.

- scikit-learn: For machine learning algorithms and utilities.

- SciPy: For sparse matrix operations.

---

## 📂 Dataset Requirements

- Ensure the following files are available in the project directory:

- train.csv: Contains training data with columns text and emotion.

- test.csv: Contains test data with columns text and emotion.

- Text_Emotion.csv: Additional dataset for merging and normalization.

- Emotion Mapping

- The emotions are mapped as follows:

- ☹️: Sad (1)

- 🙂: Happy (2)

- 😠: Angry (0)

- 😐: Neutral (3)

---

## ⚙️ How It Works

- Step 1: Dataset Preprocessing

Loads datasets and checks for missing files or errors.

Harmonizes and normalizes labels using a predefined emotion map.

Combines all datasets into a single, clean format.

- Step 2: Text Transformation

Converts text into numerical features using TF-IDF Vectorization.

Reduces feature space to 10,000 dimensions and removes stopwords.

- Step 3: Model Training

Splits data into training and test sets.

Computes dynamic class weights for handling imbalanced data.

Trains a Logistic Regression Model with up to 1,000 iterations.

- Step 4: Model Evaluation

Predicts on the test set and evaluates using:

Accuracy Score

Classification Report (Precision, Recall, F1-Score)

- Step 5: Feedback Loop

Predicts the emotion of user-provided text.

Allows users to correct predictions and retrain the model with new examples.

Emphasizes feedback examples during retraining to prioritize learning.

---

## 🛠️ Setup Instructions

- Clone the Repository:

git clone https://github.com/Tanish141/emotion-classifier.git
cd emotion-classifier

- Install Dependencies:

pip install pandas numpy scikit-learn scipy

Ensure Dataset Availability:
Ensure the following files are in the project directory:

train.csv

test.csv

Text_Emotion.csv

- Run the Script:

python emotion_classifier.py

Interactive Inference:
Enter sentences to predict emotions, provide feedback, and dynamically improve the model.

---

## 📊 Results

- The model is evaluated on the test set, providing:

- Accuracy: Measures overall prediction correctness.

- Classification Report: Includes precision, recall, and F1-score for each emotion class.

- Sample Evaluation:

Accuracy: 85%
Classification Report:
               precision    recall  f1-score   support

      Angry       0.87      0.80      0.83       150
        Sad       0.84      0.89      0.86       200
      Happy       0.88      0.90      0.89       220
     Neutral       0.80      0.78      0.79       130

    accuracy                           0.85       700
   macro avg       0.85      0.84      0.84       700
weighted avg       0.85      0.85      0.85       700

---

## 🏅 Key Features and Benefits

- Real-Time Feedback: The model learns dynamically, improving its predictions based on user input.

- Error Handling: Robust error handling for missing files and incorrect input formats.

- Scalable Training: Supports incremental learning with sparse matrix stacking.

---

## 🤝 Contributions

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request to improve the project. 🌟

---

## 📧 Contact

For any queries or suggestions, feel free to reach out:

Email: mrtanish14@gmail.com

GitHub: https://github.com/Tanish141

---

## 🎉 Happy Coding!

