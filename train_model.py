import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)              # remove numbers
    text = re.sub(r'[^\w\s]', '', text)          # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()     # remove extra spaces
    return text

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv("dataset1.csv")

df['text_'] = df['text_'].apply(clean_text)

X = df['text_']
y = df['label']

# -----------------------------
# 3. TF-IDF Vectorizer
# (Keeping strong configuration)
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(X)

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 5. Logistic Regression Model
# (Balanced to reduce bias)
# -----------------------------
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    C=1.5,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Save Model
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")