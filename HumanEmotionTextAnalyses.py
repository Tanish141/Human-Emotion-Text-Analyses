import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import vstack
import os

# Step 1: Load Datasets with enhanced error handling
data_train_path = 'train.csv'
data_test_path = 'test.csv'
data_emotion_path = 'Text_Emotion.csv'

datasets = {
    "train": data_train_path,
    "test": data_test_path,
    "emotion": data_emotion_path
}

loaded_datasets = {}
for key, path in datasets.items():
    if not os.path.exists(path):
        print(f"File not found: {path}")
    else:
        try:
            loaded_datasets[key] = pd.read_csv(path)
            print(f"Successfully loaded: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")

if len(loaded_datasets) != len(datasets):
    print("Not all datasets were loaded successfully. Please check the file paths and contents.")
    exit()

data_train = loaded_datasets.get("train", pd.DataFrame())
data_test = loaded_datasets.get("test", pd.DataFrame())
data_emotion = loaded_datasets.get("emotion", pd.DataFrame())

# Combine datasets and normalize labels
emotion_map = {"☹️": 1, "🙂": 2, "😠": 0, "😐": 3}  # Map emojis to Angry, Sad, Happy, Neutral

# Harmonize all datasets into one format
for dataset in [data_train, data_test, data_emotion]:
    if 'emotion' in dataset.columns:
        dataset['emotion'] = dataset['emotion'].map(emotion_map)
    else:
        print(f"Dataset is missing 'emotion' column. Skipping: {dataset}")
        continue

# Drop any rows with unmapped or missing values
data_combined = pd.concat([data_train, data_test, data_emotion], ignore_index=True).dropna()

if data_combined.empty:
    print("No valid data available after preprocessing. Exiting.")
    exit()

# Step 2: Prepare data for training
texts = data_combined['text'].values
labels = data_combined['emotion'].values.astype(int)  # Ensure labels are integers

# Transform text data using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42, stratify=labels)

# Compute class weights dynamically based on the actual classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {int(cls): weight for cls, weight in zip(np.unique(labels), class_weights)}

# Step 3: Build Logistic Regression Model
model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)

# Dynamically map target names based on unique labels
emotion_map_reverse = {0: "Angry", 1: "Sad", 2: "Happy", 3: "Neutral"}
dynamic_target_names = [emotion_map_reverse[label] for label in np.unique(y_test)]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=dynamic_target_names))

# Step 5: Inference with Feedback Loop
def predict_emotion(text):
    try:
        text_tfidf = tfidf_vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        return emotion_map_reverse[prediction]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"

# Store feedback dataset with corrected labels
feedback_dataset = {}

def predict_emotion_with_feedback(text):
    # Check if the input text exists in the feedback dataset
    if text in feedback_dataset:
        corrected_label = feedback_dataset[text]
        return emotion_map_reverse[corrected_label]
    else:
        # Predict using the model
        text_tfidf = tfidf_vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        return emotion_map_reverse[prediction]

def retrain_model(user_text, correct_label):
    global X_train, y_train, model
    # Store the correction in feedback dataset
    feedback_dataset[user_text] = correct_label
    
    # Add the corrected example to the training data
    new_text_tfidf = tfidf_vectorizer.transform([user_text])
    X_train = vstack([X_train, new_text_tfidf])  # Use sparse stacking
    y_train = np.append(y_train, correct_label)
    
    # Retrain the model with feedback prioritized
    for _ in range(5):  # Repeat the feedback example to emphasize importance
        X_train = vstack([X_train, new_text_tfidf])
        y_train = np.append(y_train, correct_label)
    
    model.fit(X_train, y_train)  # Retrain the model

# Real-time input loop
while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    try:
        # Use feedback-aware prediction
        emotion = predict_emotion_with_feedback(user_input)
        print(f"Predicted Emotion: {emotion}\n")

        feedback = input(f"Is this correct? (yes/no): ").strip().lower()
        if feedback == 'no':
            correct_emotion = input("Enter the correct emotion (Angry/Sad/Happy/Neutral): ").strip()
            correct_label = {"angry": 0, "sad": 1, "happy": 2, "neutral": 3}.get(correct_emotion.lower())
            if correct_label is not None:
                retrain_model(user_input, correct_label)
                print("Thank you! The model has been updated with your feedback.\n")
            else:
                print("Invalid emotion provided. No changes made.\n")
    except Exception as e:
        print(f"Error: {e}. Please try again with a valid input.\n")