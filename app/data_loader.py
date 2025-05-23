import pandas as pd
import streamlit as st

@st.cache_data
def load_books_data():
    df = pd.read_csv("enhancing/items_enhanced_final.csv")
    return df

@st.cache_data
def load_recommendations():
    df = pd.read_csv("app/hybrid_recommendations.csv")
    return df.set_index("user_id")["recommendation"].to_dict()

@st.cache_data
def load_interactions_data():
    df = pd.read_csv("data/interactions_train.csv")
    return df
