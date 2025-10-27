# 🎬 RAG-Based Movie Recommendation System

An intelligent movie recommendation system leveraging **Retrieval Augmented Generation (RAG)** with **Groq API** to deliver personalized, explainable movie suggestions through natural language queries.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This project implements a state-of-the-art movie recommendation system that combines semantic search, RAG architecture, and explainable AI to understand natural language queries and provide contextual recommendations with detailed explanations.

Unlike traditional collaborative filtering, this system understands queries like *"I want intense action movies similar to John Wick with great fight choreography"* and provides relevant recommendations backed by reasoning.

## ✨ Features

- 🔍 **Semantic Search**: Natural language understanding using sentence transformers
- 🤖 **AI-Powered Recommendations**: Groq's Llama 3.3 70B for intelligent suggestions
- 💾 **Vector Database**: ChromaDB for efficient similarity search at scale
- ⚡ **CUDA Acceleration**: GPU-accelerated embedding generation
- 🌐 **Interactive UI**: Clean Streamlit web interface
- 📊 **Multi-Dataset Integration**: 75,000+ movies from MovieLens, Netflix, and TMDB
- 📝 **Explainable Results**: Detailed reasoning for each recommendation
- 🔄 **Batch Processing**: Efficient handling of large-scale data

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | PyTorch with CUDA |
| **Embeddings** | Sentence Transformers (all-mpnet-base-v2) |
| **Vector Database** | ChromaDB |
| **LLM API** | Groq (Llama 3.3 70B Versatile) |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |

## 🏗️ System Architecture

User Query → Embedding Generation → Vector Search (ChromaDB) → Top-K Retrieval → Groq LLM → Personalized Recommendations

**Pipeline Stages:**

1. **Data Integration**: Combine MovieLens, Netflix, and TMDB datasets
2. **Embedding Generation**: Convert movies to semantic vectors using CUDA
3. **Vector Storage**: Store in ChromaDB for fast similarity search
4. **Query Processing**: User query → embedding → semantic search
5. **Generation**: Groq LLM generates explanations and recommendations

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Groq API key ([Get one here](https://console.groq.com/))

### Step 1: Clone the Repository

git clone https://github.com/yourusername/rag-movie-recommender.git
cd rag-movie-recommender

text

### Step 2: Create Virtual Environment

python -m venv .venv

text

**Activate:**
- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`

### Step 3: Install Dependencies

pip install -r requirements.txt

text

### Step 4: Download Datasets

Place the following datasets in `data/raw/`:

**MovieLens 25M**
- Download: https://grouplens.org/datasets/movielens/25m/
- Extract to: `data/raw/ml-25m/`

**Netflix Titles**
- Download: [Kaggle - Netflix Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- Place: `data/raw/netflix/netflix_titles.csv`

**TMDB 5000**
- Download: [Kaggle - TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Place: `data/raw/tmdb/tmdb_5000_movies.csv`

**Expected structure:**
data/raw/
├── ml-25m/
│ ├── movies.csv
│ ├── ratings.csv
│ └── tags.csv
├── netflix/
│ └── netflix_titles.csv
└── tmdb/
├── tmdb_5000_movies.csv
└── tmdb_5000_credits.csv

text

### Step 5: Configure Environment

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here
CUDA_VISIBLE_DEVICES=0

text

Get your Groq API key from: https://console.groq.com/

## 🚀 Usage

### Run the Application

streamlit run app.py

text

The app will open in your browser at `http://localhost:8501`

### First-Time Setup

On the **first run**, the system will:
1. Load and combine all datasets (~2-5 minutes)
2. Generate semantic embeddings (~20-30 minutes with CUDA)
3. Create ChromaDB vector database (~5 minutes)

### Subsequent Runs

After initial setup, the app loads pre-computed data in **~30 seconds**.

### Example Queries

- *"I want action movies similar to John Wick"*
- *"Recommend sci-fi movies about artificial intelligence"*
- *"Looking for heartwarming family movies"*
- *"Movies about time travel but more philosophical than action-oriented"*





## 📊 Datasets

### MovieLens 25M
- **Size**: 25 million ratings, 62,000 movies
- **Source**: GroupLens Research
- **Usage**: User ratings and movie metadata

### Netflix Titles
- **Size**: ~8,000 titles
- **Source**: Netflix catalog (Kaggle)
- **Usage**: Streaming content descriptions

### TMDB 5000
- **Size**: 5,000 movies
- **Source**: The Movie Database
- **Usage**: Detailed metadata, cast, crew, genres

**Total**: 75,000+ unique movies after deduplication

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

Model settings
models:
embedding_model: "sentence-transformers/all-mpnet-base-v2"
groq_model: "llama-3.3-70b-versatile"

RAG settings
rag:
top_k: 15
similarity_threshold: 0.5

ChromaDB settings
chroma:
collection_name: "movie_embeddings"
persist_directory: "data/processed/chroma_db"

text

## 🔬 How It Works

### 1. Data Preprocessing
- Parse JSON genres from TMDB
- Standardize genre formats across datasets
- Remove duplicates while preserving diversity
- Create combined text (title + genres + description)

### 2. Embedding Generation
- Use `sentence-transformers/all-mpnet-base-v2`
- Convert each movie to 768-dimensional vector
- CUDA acceleration for 10x speedup
- Normalize embeddings for better similarity

### 3. Vector Database
- Store embeddings in ChromaDB
- Batched insertion (5,000 items/batch) for scalability
- Cosine similarity for semantic matching

### 4. Query Processing
- Convert user query to embedding
- Search ChromaDB for top-15 similar movies
- Semantic matching, not keyword matching

### 5. Recommendation Generation
- Send retrieved movies + query to Groq LLM
- Generate 5-7 personalized recommendations
- Include detailed explanations for each suggestion

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **MovieLens** dataset by [GroupLens Research](https://grouplens.org/)
- **TMDB** for comprehensive movie metadata
- **Groq** for fast LLM inference API
- **Sentence Transformers** library by UKPLab
- **ChromaDB** for vector database capabilities

## 📧 Contact

Project Link: [https://github.com/thomson2132/rag_movie_recommender
](https://github.com/thomson2132/rag_movie_recommender.git
)

---

⭐ **If you find this project useful, please give it a star!** ⭐