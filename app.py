import streamlit as st
from src.recommender import RAGMovieRecommender
from src.utils import setup_logging
import os

logger = setup_logging()


def main():
    st.set_page_config(
        page_title="AI Movie Recommender",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 AI-Powered Movie Recommendation System")
    st.markdown("### Using RAG + Groq API for Personalized Recommendations")

    # Initialize recommender
    if 'recommender' not in st.session_state:
        with st.spinner("Initializing recommendation system..."):
            st.session_state.recommender = RAGMovieRecommender()

            # Check if system needs setup
            embeddings_path = "data/processed/embeddings.pkl"
            chroma_path = "data/processed/chroma_db"

            # If embeddings don't exist OR ChromaDB doesn't exist, run full setup
            if not os.path.exists(embeddings_path) or not os.path.exists(chroma_path):
                st.info("Setting up system for the first time...")
                st.session_state.recommender.setup_system()
                st.success("✓ Setup complete!")
            else:
                # Load existing data without recreating embeddings
                st.info("Loading existing data...")
                import pandas as pd
                from src.embeddings import EmbeddingGenerator
                from src.utils import setup_environment, load_config

                config = load_config()
                device = setup_environment()

                # Load movie data
                st.session_state.recommender.movie_data = pd.read_csv(
                    embeddings_path.replace('embeddings.pkl', 'combined_movies.csv'))

                # Load embeddings
                embedding_gen = EmbeddingGenerator(config, device)
                embeddings, movie_data = embedding_gen.load_embeddings(embeddings_path)

                # Recreate ChromaDB collection with batching
                st.session_state.recommender.retriever.create_collection(embeddings, movie_data)

                st.success("✓ System ready!")

    recommender = st.session_state.recommender

    # Main interface
    user_query = st.text_area(
        "What kind of movies are you looking for?",
        placeholder="E.g., 'I want action movies similar to John Wick' or 'Recommend romantic comedies like When Harry Met Sally'",
        height=100
    )

    if st.button("Get Recommendations", type="primary"):
        if user_query:
            with st.spinner("Finding perfect movies for you..."):
                try:
                    # Call recommend without user_id and collaborative filtering
                    recommendations, retrieved_movies = recommender.recommend(
                        user_query,
                        user_id=None,
                        use_collaborative=False
                    )

                    # Check if recommendations were generated
                    if recommendations and recommendations != "None":
                        # Display recommendations
                        st.markdown("## 🎯 Your Personalized Recommendations")
                        st.markdown(recommendations)
                    else:
                        st.error("⚠️ Failed to generate recommendations. Please check your Groq API key and try again.")
                        st.info(
                            "**Debug Info:** The system retrieved relevant movies but couldn't generate AI recommendations.")

                    # Display retrieved movies
                    with st.expander("📚 Movies Retrieved from Database", expanded=True):
                        if retrieved_movies:
                            for i, movie in enumerate(retrieved_movies[:10], 1):
                                st.write(f"{i}. **{movie['movie_title']}** - {movie.get('genres', 'N/A')}")
                        else:
                            st.warning("No movies were retrieved from the database.")

                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {e}")
                    logger.error(f"Recommendation error: {e}")

                    # Show helpful debug info
                    st.info("""
                    **Possible issues:**
                    - Groq API key might be invalid or expired
                    - Network connectivity issues
                    - API rate limit exceeded

                    Please check your `.env` file and ensure GROQ_API_KEY is set correctly.
                    """)
        else:
            st.warning("Please enter your movie preferences")


if __name__ == "__main__":
    main()
