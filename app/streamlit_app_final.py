import streamlit as st
from data_loader import load_books_data, load_recommendations, load_interactions_data
import pandas as pd

st.set_page_config(layout="wide")

# Load data outside of conditional blocks
df_books = load_books_data()
interactions = load_interactions_data()
recommendations_dict = load_recommendations()

df_books["i"] = df_books["i"].astype(str)
interactions["i"] = interactions["i"].astype(str)

# Initialize session states for show_recommendations and checkout_done
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'checkout_done' not in st.session_state:
    st.session_state.checkout_done = False
if 'borrowed_books' not in st.session_state:
    st.session_state.borrowed_books = []

# --- AUTHENTICATION STEP ---
if 'customer_authenticated' not in st.session_state:
    st.session_state.customer_authenticated = False
    st.session_state.selected_customer = None

if not st.session_state.customer_authenticated:
    st.title("Welcome to the Book Recommender System")
    st.subheader("Please enter your customer number to continue:")

    customer_list = interactions["u"].unique().tolist()

    with st.form("login_form"):
        selected_customer = st.selectbox("Select your customer number", sorted(customer_list))
        login_button = st.form_submit_button("Login")

        if login_button:
            st.session_state.selected_customer = selected_customer
            st.session_state.customer_authenticated = True
            st.rerun()  # force rerun immediately

else:
    # Logout form
    with st.form("logout_form"):
        st.write(f"Logged in as customer: **{st.session_state.selected_customer}**")
        logout_button = st.form_submit_button("Logout")

        if logout_button:
            st.session_state.customer_authenticated = False
            st.session_state.selected_customer = None
            st.session_state.borrowed_books = []
            st.rerun()  # force rerun immediately

    selected_customer = st.session_state.selected_customer

    tab1, tab2, tab3 = st.tabs(["📚 Recommender", "🛒 Borrow Basket","📊 About & EDA"])

    with tab1:
        st.title("📚 Book Recommender System")
        st.header("Welcome to the recommender system of the Cantonal Library of Vaud")
        st.subheader("Project by Amélie Madrona & Linne Verhoeven")

        st.markdown(f"Logged in as customer: **{selected_customer}**")

        shuffle_clicked = st.button("🔀 Shuffle Previously Read Books")

        if 'recent_reads' not in st.session_state or shuffle_clicked:
            st.session_state.recent_reads = (
                interactions[interactions["u"] == selected_customer]
                .merge(df_books, on="i", how="left")
                .drop_duplicates(subset=["i"])
                .sample(10, replace=False)
            )

        st.subheader("Previously borrowed books 📚")

        read_books = st.session_state.recent_reads["title_clean"].dropna().tolist()

        if not read_books:
            st.info("No previously read books found for this customer.")
        else:
            cols = st.columns(5)
            for i, title in enumerate(read_books):
                book = df_books[df_books["title_clean"] == title]
                if not book.empty:
                    book = book.iloc[0]
                    with cols[i]:
                        st.markdown(f"""
                            <div style='height: 70px; overflow: hidden; text-align: center; font-weight: bold;'>
                                {title}
                            </div>
                        """, unsafe_allow_html=True)

                        img_url = book["image"]
                        image_height = 500

                        if pd.notnull(img_url):
                            st.markdown(f"""
                                <div style="height:{image_height}px; display:flex; align-items:center; justify-content:center; margin-bottom:8px;">
                                    <img src="{img_url}" style="max-height:{image_height}px; object-fit:contain;" />
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div style='height:{image_height}px; display:flex; align-items:center; justify-content:center;
                                            text-align:center; border:1px solid #ccc; border-radius:8px;
                                            background-color:#f9f9f9; padding:10px; color:black; margin-bottom:8px;'>
                                    No image available
                                </div>
                            """, unsafe_allow_html=True)

                        with st.expander("📖 More Info"):
                            st.markdown(f"""
                            **Author**: {book['author_clean'] if pd.notnull(book['author_clean']) else 'Unknown'}

                            **Published**: {book['PublishedDate'] if pd.notnull(book['PublishedDate']) else 'N/A'}  

                            **Language**: {book['Language'] if pd.notnull(book['Language']) else 'N/A'}
                            
                            **Description**:  
                            {book['Description'] if pd.notnull(book['Description']) else 'No description available for this book.'}

                            **Subjects**: {book['Subjects'] if pd.notnull(book['Subjects']) else 'N/A'}
    
                            **Link**: {"[🔗 View on Google Books](" + book['CanonicalLink'] + ")" if pd.notnull(book.get('CanonicalLink')) else '_Not available_'}
                            """)

                        # Borrow Again button BELOW the expander
                        if book['i'] not in st.session_state.borrowed_books:
                            if st.button(f"📚 Borrow again: {title}", key=f"borrow_again_{book['i']}",use_container_width=True):
                                st.session_state.borrowed_books.append(book['i'])
                                st.success(f"Added '{title}' to your borrow basket!")

        # Show Recommendations button only AFTER login
        if st.button("Show Recommendations"):
            st.session_state.show_recommendations = True

        if st.session_state.show_recommendations:
            st.subheader("Recommended books 📖")
            recommended_ids = recommendations_dict.get(selected_customer, "").split(" ")
            recommended_books = df_books[df_books["i"].isin(recommended_ids)]

            if recommended_books.empty:
                st.warning("No recommendations available for this customer.")
            else:
                for row in range(0, len(recommended_books), 5):
                    cols = st.columns(5)
                    for i, (_, book) in enumerate(recommended_books.iloc[row:row+5].iterrows()):
                        with cols[i]:
                            st.markdown(f"""
                                <div style='height: 70px; overflow: hidden; text-align: center; font-weight: bold;'>
                                    {book['title_clean']}
                                </div>
                            """, unsafe_allow_html=True)

                            img_url = book["image"]
                            image_height = 500

                            if pd.notnull(img_url):
                                st.markdown(f"""
                                    <div style="height:{image_height}px; display:flex; align-items:center; justify-content:center; margin-bottom:8px;">
                                        <img src="{img_url}" style="max-height:{image_height}px; object-fit:contain;" />
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div style='height:{image_height}px; display:flex; align-items:center; justify-content:center;
                                                text-align:center; border:1px solid #ccc; border-radius:8px;
                                                background-color:#f9f9f9; padding:10px; color:black; margin-bottom:8px;'>
                                        No image available
                                    </div>
                                """, unsafe_allow_html=True)

                            with st.expander("📖 More Info"):
                                st.markdown(f"""
                                **Author**: {book['author_clean'] if pd.notnull(book['author_clean']) else 'Unknown'}

                                **Published**: {book['PublishedDate'] if pd.notnull(book['PublishedDate']) else 'N/A'}  
                                
                                **Language**: {book['Language'] if pd.notnull(book['Language']) else 'N/A'}

                                **Description**:  
                                {book['Description'] if pd.notnull(book['Description']) else 'No description available for this book.'}

                                **Subjects**: {book['Subjects'] if pd.notnull(book['Subjects']) else 'N/A'}
    
                                **Link**: {"[🔗 View on Google Books](" + book['CanonicalLink'] + ")" if pd.notnull(book.get('CanonicalLink')) else '_Not available_'}
                                """)

                            # Borrow button BELOW the expander
                            if book['i'] not in st.session_state.borrowed_books:
                                if st.button(f"📚 Borrow: {book['title_clean']}", key=f"borrow_{book['i']}",use_container_width=True):
                                    st.session_state.borrowed_books.append(book['i'])
                                    st.success(f"Added '{book['title_clean']}' to your borrow basket!")

    with tab2:
        st.title("🛒 Borrow Basket")
        if st.session_state.borrowed_books:
            borrowed_books_df = df_books[df_books["i"].isin(st.session_state.borrowed_books)]

            for _, book in borrowed_books_df.iterrows():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"### {book['title_clean']}")
                    img_url = book["image"]
                    if pd.notnull(img_url):
                        st.image(img_url, use_container_width=False, width=150)
                    else:
                        st.write("No image available")

                    st.markdown(f"**Author**: {book['author_clean'] if pd.notnull(book['author_clean']) else 'Unknown'}")
                    st.markdown(f"**Published**: {book['PublishedDate'] if pd.notnull(book['PublishedDate']) else 'N/A'}")

                with col2:
                    if st.button("❌ Remove", key=f"remove_{book['i']}"):
                        st.session_state.borrowed_books.remove(book['i'])
                        st.rerun()

                st.markdown("---")

            if st.button("✅ Checkout"):
                st.session_state.borrowed_books.clear()
                st.balloons()
                st.session_state.checkout_done = True
                st.rerun()

        else:
            st.info("Your borrow basket is empty.")

        if st.session_state.checkout_done:
            st.success("Thank you for borrowing! Your basket is now empty.")
            st.balloons()
            # Reset flag so message shows only once
            st.session_state.checkout_done = False

    with tab3:
        st.title("📊 About This Recommender")
        st.markdown("### 🔍 Overview")
        st.write(
            """We developed a book recommendation system using **collaborative filtering**, **content-based filtering**, and a **hybrid approach**, leveraging both interaction data and enriched metadata. The best performing model was the hybrid one, with a combination of a collaborative and 3 content-based filtering models. Models were evaluated using **Precision@10** and **Recall@10**."""
        )
        st.markdown("### 🧠 1. Recommendation Algorithms")

        st.markdown(
            """
        We implemented several recommendation algorithms to enhance the user experience:

        - **Collaborative Filtering**: Utilizes user-item interactions to recommend items.
        - **Content-Based Filtering**: Recommends items similar to those a user liked in the past.
        - **Hybrid Model**: Combines both approaches for improved accuracy.

        If you're interested in how these algorithms work, please check the sections below.
        """
        )

        with st.expander("1. Collaborative Filtering (User/User & Item/Item)"):
            st.markdown("""
            **User-Based CF:**
                        
            - **Concept**: Recommend books liked by users who are similar to the target user.
                        
            - **Baseline similarity**: Cosine similarity: measures the angle between user vectors; suitable for sparse, implicit data.
                        
            - **K-Nearest Neighbors (KNN)**: With the goal of improving and evaluating the User-based collaborative filtering recommender system, we implemented a KNN-based variant using scikit-learn’s `NearestNeighbors`. Instead of relying on a full similarity matrix (which is rather computationally heavy and noisy), the knn approach identifies only the top-*k* most similar items for each prediction. We tested multiple *k* values (ranging from 10 to 100 neighbors) using *5* randomized train-test splits. For each configuration, we measured the mean *Precision@10*. We visualized the results with error bars to reflect performance variability across different random splits [see graph]. We found optimal performance at **k = 70**. 
                        
            - **Conclusion**: Cosine similarity consistently outperformed other metrics for item-item collaborative filtering in our implicit feedback setting.

            **Item-Based CF:**
                        
            - **Concept**: Recommend books similar to those a user already interacted with.
                        
            - **Baseline Similarity**: Cosine similarity, which measures the angle between item vectors; suitable for sparse, implicit data.
                        
            - **K-Nearest Neighbors (KNN)**: As in the User-Based CF case, we tested again different values for k (number of neighbors) and found optimal performance at-*k = 70*-.
                        
            - **Pearson Correlation**: Not used because it's more effective for **explicit ratings** (e.g., a book rating from 1–5). Pearson correlation adjusts for user bias.
                        
            - **Conclusion**: Cosine similarity consistently outperformed other metrics for item-item collaborative filtering in our implicit feedback setting ([This is in line with academic literature](https://link.springer.com/chapter/10.1007/978-981-10-7398-4_37)).
            """)

        with st.expander("2. Content-Based Filtering"):
            st.markdown("""
            **TF-IDF**:
            - **What**: This is a classic method in information retrieval. TF-IDF breaks down text into individual tokens and measures word importance relative to all other books.
            - **How**: TF-IDF represents text as sparse vectors based on word frequency, adjusted by how unique each word is.
            - **Use Case**: Good for surface-level textual similarities (e.g., shared keywords).
            - **Example**: Book: *Harry Potter and the Philosopher's Stone*, Author: *J.K. Rowling*, Publisher: *Bloomsbury*  
             TF-IDF counts the frequency of each word, downweights common ones like “publishing,” and generates a sparse vector.

            **BERT**:
            - **What**: Deep learning model (transformer architecture) that takes full phrases or sentences.
            - **How**: Generates dense, contextualized embeddings that understand semantic meaning.
            - **Use Case**: Captures deeper relationships in content (e.g., plot similarities).
            - **Example**: With the input: “Harry Potter and the Philosopher's Stone J.K. Rowling Bloomsbury” BERT understands context and recognizes title, author, and organization even without exact matches.

            **Google Embeddings API**:
           - **What**: The `gemini-embedding-001` model from Google, accessed via API.
           - **How**: Uses pretrained transformer models like BERT, but more advanced.
           - **Use Case**: Leading semantic embedding model ([MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)). Easy integration and efficient.""")

        with st.expander("3. Hybrid Model"):
            st.markdown("""
            We combined both collaborative and content-based approaches using a **weighted sum** of different similarity matrices. We did this to leverage the strengths of each system while mitigating their individual weaknesses. Collaborative filtering captures patterns from user behavior but struggles with new or sparsely rated items, while content-based filtering can recommend new or niche items using item attributes but often lacks diversity. By blending them together, we ensure that recommendations remain accurate even when user interaction data is limited, while also introducing semantic richness and personalization based on content.
            
            Best performing combined similarities:
            ```python
            hybrid_sim = a * tfidf_sim + b * item_cf_sim + c * google_sim + d * bert_sim
            ```
            - We tuned weights using a simplified grid search
        """)

        st.markdown("---")
        st.markdown("### 📊 Evaluation Metrics")

        # Replace with your actual values
        st.markdown("""**Evaluation Metrics**: Here is an overview of the Precision@10 and Recall@10 metrics that evaluate the performance of our recommendation algorithms. These metrics help us understand how well our models are performing in terms of recommending relevant items to users. Precision@10 indicates the proportion of recommended items that are relevant out of all recommended items, while Recall@10 indicates the proportion of relevant items that are recommended out of all relevant items. A higher value for both metrics indicates better performance.""")
        table_data = {
            "Model": [
                "User-User CF",
                "Item-Item CF",
                "TF-IDF (Content)",
                "BERT (Content)",
                "Google Gemini Embeddings",
                "Hybrid (CF + BERT + Google)",
                "Hybrid (CF + TF-IDF + Google)",
                "Hybrid (CF + TF-IDF + BERT + Google)"
            ],
            "Precision@10": [
                "0.0612",
                "0.0585",
                "0.0149",
                "0.1760",
                "0.0480",
                "0.0630",
                "0.0630",
                "0.0623"
            ],
            "Recall@10": [
                "0.3167",
                "0.2820",
                "0.0910",
                "0.1760",
                "0.2700",
                "0.2990",
                "0.3000",
                "0.3220"
            ]
        }
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧪 2. Overview of our training data")

        st.subheader("Interactions Dataset")
        st.markdown("""The data we use for this recommender system is the very interactions that our library had with, you its customers! Here are a few statistics to understand the data we are working with:
        - **Total Interactions**: 87,047  
        - **Unique Users**: 7,838  
        - **Unique Items**: 15,291  
        - **Average Interactions/User**: 11  
        - **Median Interactions/User**: 6  
        - Distribution is positively skewed (up to 385 interactions/user, you go, book worm!).
        """)

        # Show image of interactions per user
        st.image("readme_images/distribution_interactions_per_user.png", caption="Distribution of Interactions per User")

        st.subheader("Items Dataset")
        st.markdown("""
        - Initial missing values: ~5% ISBNs, 15% Authors, 17% Subjects.
        - Metadata cleaned:
        - Extracted first valid ISBN
        - Removed slashes from titles
        - Cleaned author names
        """)

        st.image("readme_images/non_missing_data_plot.png", caption="Original vs. Enhanced Non-Missing Data")

        st.markdown("### 🔬 Data Enhancing")
        st.markdown("""Here are the steps we took to enhance our data and make it more useful for our recommender system:
        **Using Google Books API & ISBNDB:**
        - Filled missing entries: Description, Publisher, Subjects, Language, Pub Date
        - Added links: Google Canonical link & Book Cover
        - Priority order for merging: Original > Google > ISBNDB

        **BERTopic Topic Modeling:**
        - Extracted 25 semantic topics using BERT + UMAP
        """)

        st.image("readme_images/reduced_topic_distribution.png", caption="Reduced Topic Distributions")

        st.markdown("### 💬 Embeddings")
        st.markdown("""Behind our top-performing models, we used embeddings to capture the meaning of the text. Embeddings are numerical representation of texts (vectors) that capture meaning, and in the case of BERT and the Gemini model, also capture context. Here are the details of the embedding models we used:
        **Using Google API & BERT:**
        - Generated embeddings for a combination of book title, descriptions, authors, publication date, and subjects. This trial and error approach was used to find the best combination of features to generate the embeddings that would capture meaning in the most relevant way for our recommender system.
        - Captured semantic meaning and context
        - Enhanced content-based recommendations by incorporating meaning through embeddings
        """)
