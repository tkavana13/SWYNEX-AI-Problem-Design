import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/support_tickets.csv")
df["text"] = df["subject"].fillna("") + " " + df["description"].fillna("")

X = df["text"]
y = df["priority"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Classification report:")
print(classification_report(y_test, pred, digits=3))

print("Confusion matrix:")
print(confusion_matrix(y_test, pred, labels=["P1", "P2", "P3"]))

examples = [
    "Production database is unavailable for all users",
    "How do I change my email notification settings?",
    "The monthly report export is failing for my account"
]

print("\nExample predictions:")
for text in examples:
    label = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    confidence = probs.max()
    print(f"{label} | confidence={confidence:.3f} | {text}")
