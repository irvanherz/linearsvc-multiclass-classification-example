from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tabulate import tabulate

df = pd.read_csv('comments.csv')
df.head()

label_ids = df.drop_duplicates().label.value_counts().to_numpy()
df['label_id'] = df['label'].factorize()[0]
# print(tabulate(df, headers='keys', tablefmt='psql'))


X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['label_id'])

vectorizer = TfidfVectorizer(
    stop_words="english", max_features=1000, decode_error="ignore")
vectorizer.fit(X_train)

cls = LinearSVC()
cls.fit(vectorizer.transform(X_train), y_train)

y_pred = cls.predict(vectorizer.transform(X_test))
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


samples = [
    "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free!",
    "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight",
    "I‘m going to try for 2 months ha ha only joking"
]

for i, j in enumerate(samples):
  cat_id = cls.predict(vectorizer.transform([j]))
  print(cat_id)
