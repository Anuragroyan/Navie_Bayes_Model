
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Sample dataset (you can expand it)
data = {
    "text": [
        "Yeah right, that was amazing",      
        "I love spending time in traffic",    
        "What a surprise, he failed again",   
        "I had a great day at the park",      
        "She is really talented",             
        "The food was delicious", 
        "I'm so happy to be stuck in traffic for two hours",
        "What a wonderful day to forget my umbrella",
        "I love when my internet stops working during meetings",
        "The food was absolutely amazing, just kidding it was terrible",
        "I won the lottery! Just kidding, I lost my wallet",
        "Looking forward to Monday... said no one ever",
        "It's raining again, how lovely",
        "The product was great and worked as expected",
        "Thank you for your help, it was really appreciated",
        "This is the best movie I’ve seen all year"                       \
    ],
    "label": [1, 1, 1, 0, 0, 0,1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  # 1 = Sarcastic, 0 = Not
}

df = pd.DataFrame(data)

# 2. Vectorize text
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 3. Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# 4. Export vocabulary and class log probabilities
export_data = {
    "vocabulary": vectorizer.vocabulary_,
    "class_log_prior": model.class_log_prior_.tolist(),
    "feature_log_prob": model.feature_log_prob_.tolist(),
    "classes": model.classes_.tolist()
}

# 5. Save to JSON
with open("naive_bayes_model.json", "w") as f:
    json.dump(export_data, f)

print("Model exported to naive_bayes_model.json")