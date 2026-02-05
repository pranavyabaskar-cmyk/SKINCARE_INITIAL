import streamlit as st
import pandas as pd
import numpy as np
import joblib
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from mistralai import Mistral

# ===============================
# 🔹 API Key & Client Setup
# ===============================
API_KEY = "dWZ79JPVVU8thW4jPLzpPv61rkCyPYy6"
client = Mistral(api_key=API_KEY)

# ===============================
# 🔹 Mistral Chatbot Function
# ===============================
def call_skincare_chatbot(user_text: str) -> str:
    prompt = f"""
<s>[INST]
You are an AI skincare advisor. Never mention brand names. 
Give concise, friendly advice based on user's skin type and concerns.
Recommend product types, key ingredients, and routine steps.
User: {user_text}
[/INST]</s>
"""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.complete(model="mistral-small-latest", messages=messages)
    return response.choices[0].message.content.strip()

# ===============================
# 🔹 Load All Models & Data
# ===============================
@st.cache_resource
def load_all_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    df = joblib.load("skincare_df.pkl")

    # SentenceTransformer + FAISS index
    model = SentenceTransformer("sentence_model")
    index = faiss.read_index("skincare_faiss.index")

    return tfidf, tfidf_matrix, model, index, df

tfidf, tfidf_matrix, sbert_model, sbert_index, df = load_all_models()

# ===============================
# 🔹 Streamlit Page Setup
# ===============================
st.set_page_config(page_title="AI Skincare System", layout="wide")

st.sidebar.title("🧴 Skincare AI System")
app_mode = st.sidebar.radio(
    "Choose a Mode:",
    ["💬 Chatbot Assistant", "📊 TF-IDF Recommender", "🧠 Transformer Recommender"]
)

# =======================================================
# 💬 MODE 1: Chatbot (Mistral AI)
# =======================================================
if app_mode == "💬 Chatbot Assistant":
    st.title("💬 AI Skincare Assistant")
    chat_input = st.text_area("Describe your skin concerns or questions:", height=150)
    if st.button("Submit"):
        if chat_input.strip():
            with st.spinner("Analyzing your skin concerns..."):
                advice_text = call_skincare_chatbot(chat_input)
                st.markdown(f"### 🧖 Skincare Advice:\n\n{advice_text}")
        else:
            st.warning("Please enter your concerns first.")

# =======================================================
# 📊 MODE 2: TF-IDF BASED RECOMMENDER
# =======================================================
elif app_mode == "📊 TF-IDF Recommender":
    st.title("🫧⋆｡˚ Skincare Product Recommender (TF-IDF) °‧🫧⋆࿔*")
    
    primary_categories = sorted(df["primary_category"].dropna().unique())
    tertiary_categories = sorted(df["tertiary_category"].dropna().unique())

    selected_primary = "Skincare"
    selected_tertiary = st.selectbox("Tertiary Category", options=["Any"] + tertiary_categories)
    user_input = st.text_area(
        "Describe your needs (ex: acne, dryness, wrinkles, alcohol free etc)", height=150
    )
    price_pref = st.selectbox("Preferred Price Range", ["Any", "Low", "Medium", "High"])
    rating_pref = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)

    def apply_filters(df_, rating_min, price_pref_, selected_tertiary_):
        filtered = df_.copy()
        if rating_min > 0:
            filtered = filtered[filtered["rating"] >= rating_min]
        if price_pref_ != "Any":
            low_q = df_["price_usd"].quantile(0.33)
            med_q = df_["price_usd"].quantile(0.66)
            if price_pref_ == "Low":
                filtered = filtered[filtered["price_usd"] <= low_q]
            elif price_pref_ == "Medium":
                filtered = filtered[(filtered["price_usd"] > low_q) & (filtered["price_usd"] <= med_q)]
            elif price_pref_ == "High":
                filtered = filtered[filtered["price_usd"] > med_q]
        if selected_tertiary_ != "Any":
            filtered = filtered[filtered["tertiary_category"] == selected_tertiary_]
        return filtered

    def get_tfidf_recommendations(user_text, rating_min, price_pref_, selected_tertiary_, top_n=3):
        if not user_text.strip():
            return pd.DataFrame()

        combined_input = " ".join([selected_tertiary_, user_text]).strip()
        user_tfidf = tfidf.transform([combined_input])
        cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)[0]
        df['sim_score'] = cosine_sim
        df_sorted = df.sort_values(by='sim_score', ascending=False)
        filtered = apply_filters(df_sorted, rating_min, price_pref_, selected_tertiary_)
        return filtered.head(top_n)

    if st.button("🔍 Get TF-IDF Recommendations"):
        if not user_input.strip():
            st.warning("Please enter your skin concerns.")
        else:
            with st.spinner("Finding best matches..."):
                recommended = get_tfidf_recommendations(
                    user_input, rating_pref, price_pref, selected_tertiary, top_n=3
                )

            if recommended.empty:
                st.error("No matches found. Try relaxing filters.")
            else:
                st.subheader("💆🏻‍♀️ Top 3 Recommended Skincare Products (TF-IDF)")
                for _, row in recommended.iterrows():
                    st.markdown(f"### 🧴 {row.get('product_name', 'Unknown')}")
                    st.markdown(f"**Brand:** {row.get('brand_name', 'N/A')}")
                    st.markdown(f"**Price:** 💰 {row.get('price_usd', 'N/A')}")
                    st.markdown(f"**Rating:** ⭐ {row.get('rating', 'N/A')}")
                    st.markdown(f"**Category:** {row.get('primary_category', 'N/A')} | {row.get('tertiary_category', 'N/A')}")
                    st.markdown(f"**Similarity Score:** {row.get('sim_score', 0):.3f}")
                    st.markdown(f"**Ingredients:** {str(row.get('ingredients', ''))[:300]}...")
                    st.divider()

# =======================================================
# 🧠 MODE 3: SentenceTransformer + FAISS Recommender
# =======================================================
else:
    st.title("💆‍♀️ Skincare Recommender — Transformer (all-MiniLM-L6-v2)")

    primary_categories = sorted(df["primary_category"].dropna().unique())
    tertiary_categories = sorted(df["tertiary_category"].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        selected_primary = st.selectbox("Primary Category", ["Any"] + primary_categories)
    with col2:
        selected_tertiary = st.selectbox("Tertiary Category", ["Any"] + tertiary_categories)

    user_input = st.text_area("Describe your skin concerns (e.g., acne, dryness, wrinkles)", height=150)
    rating_pref = st.slider("⭐ Minimum Rating", 0.0, 5.0, 3.5, 0.1)
    top_n = 5

    def get_transformer_recommendations(query, top_n=5):
        if not query.strip():
            return pd.DataFrame()
        query_vec = sbert_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        sim_scores, indices = sbert_index.search(query_vec, top_n * 5)
        results = df.iloc[indices[0]].copy()
        results["similarity"] = sim_scores[0]
        results = results[results["rating"] >= rating_pref]
        if selected_primary != "Any":
            results = results[results["primary_category"] == selected_primary]
        if selected_tertiary != "Any":
            results = results[results["tertiary_category"] == selected_tertiary]
        return results.head(top_n)

    if st.button("🔍 Get Transformer Recommendations"):
        if not user_input.strip():
            st.warning("Please describe your skin concerns.")
        else:
            with st.spinner("Searching for similar products..."):
                recommendations = get_transformer_recommendations(user_input, top_n)
            if recommendations.empty:
                st.error("No products found. Try adjusting filters.")
            else:
                st.subheader("✨ Top Recommendations (Transformer)")
                for _, row in recommendations.iterrows():
                    st.markdown(f"### 🧴 {row.get('product_name', 'Unknown Product')}")
                    st.markdown(f"**Brand:** {row.get('brand_name', 'N/A')}")
                    st.markdown(f"**Price:** ${row.get('price_usd', 'N/A')}")
                    st.markdown(f"**Rating:** ⭐ {row.get('rating', 'N/A')}")
                    st.markdown(
                        f"**Category:** {row.get('primary_category', 'N/A')} | "
                        f"**Tertiary:** {row.get('tertiary_category', 'N/A')}"
                    )
                    st.markdown(f"**Similarity Score:** {row.get('similarity', 0):.3f}")
                    ingredients_text = str(row.get('ingredients', ''))
                    if len(ingredients_text) > 300:
                        ingredients_text = ingredients_text[:300] + "..."
                    st.markdown(f"**Ingredients:** {ingredients_text}")
                    st.divider()
