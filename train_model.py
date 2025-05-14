# # import nltk
# # import pandas as pd
# # import numpy as np
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.model_selection import train_test_split
# # from sklearn.naive_bayes import MultinomialNB
# # from sklearn.metrics import accuracy_score
# # import pickle
# # import re

# # # Download required NLTK data
# # nltk.download('punkt')
# # nltk.download('stopwords')

# # # Sample data - you can replace this with your own dataset
# # data = {
# #     'text': [
# #         "I love this product, it's amazing!",
# #         "This is the worst experience ever.",
# #         "The service was okay, nothing special.",
# #         "I'm really happy with my purchase.",
# #         "Terrible customer service.",
# #         "It's a decent product.",
# #         "I absolutely hate this.",
# #         "The quality is excellent!",
# #         "Not bad, but could be better.",
# #         "I'm disappointed with the results."
# #     ],
# #     'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 
# #                  'neutral', 'negative', 'positive', 'neutral', 'negative']
# # }

# # # Create DataFrame
# # df = pd.DataFrame(data)

# # def preprocess_text(text):
# #     # Convert to lowercase
# #     text = text.lower()
# #     # Remove special characters and digits
# #     text = re.sub(r'[^a-zA-Z\s]', '', text)
# #     return text

# # # Preprocess the text
# # df['processed_text'] = df['text'].apply(preprocess_text)

# # # Create TF-IDF vectorizer
# # vectorizer = TfidfVectorizer(max_features=5000)
# # X = vectorizer.fit_transform(df['processed_text'])
# # y = df['sentiment']

# # # Split the data
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Train the model
# # model = MultinomialNB()
# # model.fit(X_train, y_train)

# # # Evaluate the model
# # y_pred = model.predict(X_test)
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"Model Accuracy: {accuracy}")

# # # Save the model and vectorizer
# # with open('model/sentiment_model.pkl', 'wb') as f:
# #     pickle.dump(model, f)

# # with open('model/vectorizer.pkl', 'wb') as f:
# #     pickle.dump(vectorizer, f)

# # print("Model and vectorizer saved successfully!")



# # import pandas as pd

# # # Sample sentiment data
# # positive_samples = [
# #     "I love this product!",
# #     "This is amazing!",
# #     "Excellent service!",
# #     "I'm so happy with the result!",
# #     "Great work, I’m impressed!"
# # ]

# # negative_samples = [
# #     "I hate this product.",
# #     "This is the worst.",
# #     "Terrible service!",
# #     "I’m not happy at all.",
# #     "This is so bad!"
# # ]

# # neutral_samples = [
# #     "It's okay.",
# #     "Nothing special.",
# #     "Average experience.",
# #     "It works, I guess.",
# #     "Not good, not bad."
# # ]

# # # Create DataFrame
# # df = pd.DataFrame({
# #     'tweet': positive_samples + negative_samples + neutral_samples,
# #     'label': (
# #         ['Positive'] * len(positive_samples) +
# #         ['Negative'] * len(negative_samples) +
# #         ['Neutral'] * len(neutral_samples)
# #     )
# # })

# # # Shuffle the data
# # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # # Features and labels
# # X = df['tweet']
# # y = df['label']

# # # Vectorization
# # vectorizer = TfidfVectorizer()
# # X_vect = vectorizer.fit_transform(X)

# # # Train/Test split
# # X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# # # Train model
# # model = LogisticRegression(max_iter=1000)
# # model.fit(X_train, y_train)

# # # Evaluation
# # y_pred = model.predict(X_test)
# # print("\nClassification Report:\n")
# # print(classification_report(y_test, y_pred))

# # # Save model
# # os.makedirs("model", exist_ok=True)
# # with open("model/sentiment_model.pkl", "wb") as f:
# #     pickle.dump(model, f)

# # with open("model/vectorizer.pkl", "wb") as f:
# #     pickle.dump(vectorizer, f)

# # print("✅ Model with Neutral sentiment trained and saved.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os

# ✅ Load CSV with encoding fix
data = pd.read_csv("test.csv", encoding='ISO-8859-1')

# ✅ Show column info
print("Columns:", data.columns)

# ✅ Remove rows with missing text or sentiment
data = data[['text', 'sentiment']]  # keep only the required columns
data = data.dropna(subset=['text', 'sentiment'])  # remove rows with NaN in these columns
data = data[data['text'].str.strip().astype(bool)]  # remove empty strings

# ✅ Print sample
print("Sample cleaned data:\n", data.head())

# ✅ Prepare features and labels
X = data['text']
y = data['sentiment']

# ✅ Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# ✅ Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model and vectorizer
os.makedirs("model", exist_ok=True)
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved to 'model/' folder.")
