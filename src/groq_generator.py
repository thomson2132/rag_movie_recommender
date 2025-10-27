from groq import Groq
from src.utils import get_groq_api_key, setup_logging
import time

logger = setup_logging()


class GroqGenerator:
    def __init__(self, config):
        self.config = config
        try:
            api_key = get_groq_api_key()
            if not api_key:
                raise ValueError("GROQ_API_KEY is empty or not set in .env file")

            self.client = Groq(api_key=api_key)
            self.model = config['models']['groq_model']
            logger.info(f"✓ Groq client initialized with model: {self.model}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq client: {e}")
            raise

    def create_recommendation_prompt(self, user_query, retrieved_movies):
        """Create improved prompt for movie recommendation."""
        movies_list = []
        for i, movie in enumerate(retrieved_movies[:10], 1):
            title = movie.get('movie_title', 'Unknown')
            genres = movie.get('genres', 'N/A')
            movies_list.append(f"{i}. {title} (Genres: {genres})")

        movies_text = "\n".join(movies_list)

        prompt = f"""You are an expert movie recommendation assistant. A user is looking for movies with these preferences:

USER REQUEST: "{user_query}"

RELEVANT MOVIES FROM DATABASE:
{movies_text}

YOUR TASK:
Based on the user's request and the retrieved movies, provide 5-7 personalized movie recommendations. 

For each recommendation:
1. Explain why it matches the user's preferences
2. Highlight key themes, actors, or style similarities
3. Mention what makes it special or worth watching

Format your response in a clear, engaging way with bullet points or numbered list. Be enthusiastic but authentic in your recommendations."""

        return prompt

    def generate_recommendations(self, user_query, retrieved_movies, max_retries=3):
        """Generate personalized recommendations using Groq with retry logic."""
        logger.info("Generating recommendations with Groq...")

        if not retrieved_movies:
            logger.warning("No movies retrieved for recommendation generation")
            return "No relevant movies found in the database. Please try a different query."

        prompt = self.create_recommendation_prompt(user_query, retrieved_movies)

        # Retry logic for API calls
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} to call Groq API")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a knowledgeable and enthusiastic movie recommendation expert. Provide personalized, detailed recommendations based on user preferences."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                    top_p=0.9
                )

                recommendation = response.choices[0].message.content

                if recommendation:
                    logger.info("✓ Successfully generated recommendations")
                    return recommendation
                else:
                    logger.error("Empty response from Groq API")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error on attempt {attempt + 1}: {error_msg}")

                # Specific error handling
                if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                    logger.error("❌ Authentication error - check your GROQ_API_KEY")
                    return None
                elif "rate_limit" in error_msg.lower():
                    logger.warning("Rate limit hit, waiting before retry...")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    logger.error("❌ Network connection error")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                else:
                    logger.error(f"❌ Unexpected error: {error_msg}")

                # If last attempt, return None
                if attempt == max_retries - 1:
                    return None

        return None
