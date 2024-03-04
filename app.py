import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def preprocess_text(text):
    # Eliminate handles and URLs
    text = ' '.join([word for word in text.split() if not word.startswith('@')])  # Remove handles
    text = ' '.join([word for word in text.split() if not word.startswith('http')])  # Remove URLs
    
    # Tokenize the string into words
    words = text.split()
    
    # Remove stop words
    stop_words = [
    "and", "is", "a", "on", "etc",
    "with", "this", "was", "the", "for",
    "I'm", "of", "at", "all", "it",
    "I", "the", "was", "I'm", "with",
    "the", "was", "I'm", "with", "and",
    "it", "I", "ever", "made", "of"
    ]
    words = [word for word in words if word.lower() not in stop_words]
    
    
    # Convert all words to lowercase
    words = [word.lower() for word in words]
    
    return ' '.join(words)

# Read data from CSV file
with open('sentiment_data.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Preprocess the data
for i in range(1, len(data)):  # Assuming the first row contains headers
    data[i][0] = preprocess_text(data[i][0])

# Split the data into features (X) and labels (y)
X = np.array(data[1:])[:, 0]
y = np.array(data[1:])[:, 1]

# Encode labels (Positive -> 1, Negative -> 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use CountVectorizer to convert text data into a matrix of token counts
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Get user input for testing
user_input = input("Enter a sentence for sentiment analysis: ")
user_input = preprocess_text(user_input)
user_input_vectorized = vectorizer.transform([user_input])

# Predict the sentiment for user input
prediction = model.predict(user_input_vectorized)

# Decode the predicted label
predicted_sentiment = label_encoder.inverse_transform(prediction)

# Display the result
#print(f"Sentiment analysis for the input: '{user_input}'")
print(f"Predicted Sentiment: {predicted_sentiment[0]}")
