import streamlit as st
import os

st.set_page_config(
    page_title="AI Movie Recommender - Demo",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI-Powered Movie Recommendation System")
st.markdown("### RAG + Groq API Demo")






# Demo interface
user_query = st.text_area(
    "What kind of movies are you looking for?",
    placeholder="E.g., 'I want action movies similar to John Wick'",
    height=100
)

if st.button("Get Demo Recommendations", type="primary"):
    if user_query:
        with st.spinner("Generating recommendations..."):
            st.markdown("## 🎯 Sample Recommendations")

            # Mock recommendations based on common queries
            if "john wick" in user_query.lower() or "action" in user_query.lower():
                st.markdown("""
                1. **John Wick: Chapter 2 (2017)** - Action | Thriller
                   - Continues the stylish, gun-fu action of the first film
                   - Similar revenge-driven narrative with underground criminal world
                   - Exceptional choreography and world-building

                2. **The Raid (2011)** - Action | Thriller | Crime
                   - Indonesian martial arts masterpiece with relentless action
                   - Intense hand-to-hand combat in a contained setting
                   - Raw, visceral fight sequences

                3. **Atomic Blonde (2017)** - Action | Thriller | Spy
                   - Stylish Cold War spy thriller with brutal fight choreography
                   - Similar aesthetic and tonal approach to John Wick
                   - Charlize Theron's intense physical performance

                4. **Extraction (2020)** - Action | Thriller
                   - High-octane action with impressive long-take sequences
                   - Chris Hemsworth in a gritty, grounded role
                   - Similar focus on practical stunts and choreography

                5. **Nobody (2021)** - Action | Thriller | Crime
                   - Bob Odenkirk as an underestimated badass
                   - Dark humor mixed with intense action
                   - Similar revenge plot with underground connections
                """)

            elif "sci-fi" in user_query.lower() or "science fiction" in user_query.lower():
                st.markdown("""
                1. **Blade Runner 2049 (2017)** - Sci-Fi | Mystery
                   - Visually stunning sequel exploring themes of humanity
                   - Thought-provoking narrative about identity and memory

                2. **Arrival (2016)** - Sci-Fi | Drama
                   - Intelligent first-contact story with linguistic focus
                   - Emotional depth combined with hard sci-fi concepts

                3. **Ex Machina (2014)** - Sci-Fi | Thriller
                   - Intimate AI exploration with philosophical depth
                   - Suspenseful and thought-provoking

                4. **Interstellar (2014)** - Sci-Fi | Adventure
                   - Epic space exploration with scientific grounding
                   - Emotional family story wrapped in cosmic adventure

                5. **The Martian (2015)** - Sci-Fi | Adventure
                   - Optimistic survival story with problem-solving focus
                   - Science-based approach to extraterrestrial challenges
                """)

            elif "comedy" in user_query.lower() or "funny" in user_query.lower():
                st.markdown("""
                1. **The Grand Budapest Hotel (2014)** - Comedy | Adventure
                   - Wes Anderson's whimsical masterpiece
                   - Visual feast with deadpan humor

                2. **Knives Out (2019)** - Comedy | Mystery | Crime
                   - Modern whodunit with sharp wit
                   - Ensemble cast in a clever, entertaining mystery

                3. **Jojo Rabbit (2019)** - Comedy | Drama | War
                   - Dark comedy with heart and important themes
                   - Taika Waititi's unique comedic voice

                4. **The Nice Guys (2016)** - Comedy | Action | Mystery
                   - Buddy cop comedy with 70s flair
                   - Russell Crowe and Ryan Gosling's chemistry

                5. **Game Night (2018)** - Comedy | Mystery
                   - Fast-paced comedy with mystery elements
                   - Great ensemble chemistry and clever plot
                """)

            else:
                st.markdown("""
                1. **Parasite (2019)** - Drama | Thriller
                   - Masterful class commentary with genre-blending brilliance

                2. **Everything Everywhere All at Once (2022)** - Action | Sci-Fi | Comedy
                   - Mind-bending multiverse adventure with emotional core

                3. **Inception (2010)** - Sci-Fi | Thriller
                   - Complex heist thriller set in dream worlds

                4. **The Shawshank Redemption (1994)** - Drama
                   - Timeless story of hope and friendship

                5. **Whiplash (2014)** - Drama | Music
                   - Intense student-teacher dynamic in music conservatory
                """)

            # Show how the full system works
            with st.expander("🔬 How The Full System Works"):
                st.markdown("""
                **Technical Architecture:**

                1. **Data Integration**: Combines MovieLens 25M, Netflix, and TMDB datasets (75,000+ movies)
                2. **Semantic Embeddings**: Uses Sentence Transformers to create 768-dimensional vectors
                3. **Vector Database**: ChromaDB stores embeddings for fast similarity search
                4. **Retrieval**: Finds top-15 semantically similar movies based on your query
                5. **Generation**: Groq's Llama 3.3 70B LLM generates personalized explanations

                **Technologies Used:**
                - Python, PyTorch, Sentence Transformers
                - ChromaDB for vector search
                - Groq API for LLM inference
                - Streamlit for web interface
                """)

            with st.expander("📚 Sample Retrieved Movies (Demo)"):
                st.info("In the full version, this shows actual movies from the vector database.")
                st.write("1. **John Wick (2014)** - Action | Thriller")
                st.write("2. **John Wick: Chapter 2 (2017)** - Action | Crime")
                st.write("3. **The Equalizer (2014)** - Action | Crime | Thriller")
                st.write("4. **Taken (2008)** - Action | Thriller")
                st.write("5. **The Raid (2011)** - Action | Thriller")

    else:
        st.warning("Please enter your movie preferences")

# Footer with links
