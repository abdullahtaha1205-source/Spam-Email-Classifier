This project focuses on building a machine learning model that can automatically detect and classify emails as Spam or Ham (Not Spam). Using Natural Language Processing (NLP) techniques and algorithms such as Naive Bayes and Support Vector Machines (SVM), the model learns patterns in email content to effectively filter out spam messages.

🔹 Features

Preprocess email text (tokenization, stopword removal, stemming/lemmatization).

Convert text into numerical features using TF-IDF Vectorization.

Train and evaluate classification models (Naive Bayes, SVM).

Evaluate performance with Accuracy, Precision, Recall, and F1-score.

Predict whether a new email is spam or ham.

🔹 Tech Stack

Python 🐍

Pandas & NumPy → Data processing

NLTK / Scikit-learn → NLP & ML models

Matplotlib / Seaborn → Performance visualization

🔹 Future Enhancements

Improve accuracy with deep learning (RNN/LSTM).

Spam Email Classifier
import pandas as pd from sklearn.model_selection import train_test_split from sklearn.feature_extraction.text import CountVectorizer from sklearn.naive_bayes import MultinomialNB from sklearn.metrics import accuracy_score, classification_report

Sample dataset (you can replace with full Kaggle dataset later)
data = { "label": ["ham", "spam", "ham", "ham", "spam"], "message": [ "Hi, how are you?", "Congratulations! You won a free lottery ticket. Call now!", "Are we meeting tomorrow?", "Don't forget to bring your notes.", "Get a loan approved instantly, apply now!" ] }

df = pd.DataFrame(data)

Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=42)

Vectorize text
vectorizer = CountVectorizer() X_train_vec = vectorizer.fit_transform(X_train) X_test_vec = vectorizer.transform(X_test)

Train Naive Bayes classifier
model = MultinomialNB() model.fit(X_train_vec, y_train)

Predictions
y_pred = model.predict(X_test_vec)

Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred)) print(classification_report(y_test, y_pred))

Test with custom messages
test_messages = ["Free entry in a contest! Win big prizes!", "Hey, are we still on for lunch?"] test_vec = vectorizer.transform(test_messages) print("Predictions (0=ham, 1=spam):", model.predict(test_vec))

Deploy as a web application (Flask/Streamlit).

Real-time spam detection for incoming emails.
