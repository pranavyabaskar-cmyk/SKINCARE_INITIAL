import joblib
from sklearn.metrics.pairwise import cosine_similarity

tfidf = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
df = joblib.load("skincare_df.pkl")

queries = [
    "vegan sunscreen",
    "anti-aging serum",
    "acne spot treatment",
    "moisturizer for dry skin"
]

for q in queries:
    q_vec = tfidf.transform([q])
    sim = cosine_similarity(q_vec, tfidf_matrix)[0]
    top = df.iloc[sim.argsort()[-3:][::-1]]
    print(f"\n🔍 Query: {q}")
    print(top[["product_name", "rating", "price_usd"]])
