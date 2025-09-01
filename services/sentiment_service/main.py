import sys
import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

# --- 1. PATH CORRECTION ---
# This is a crucial step. It adds the root directory of your project to the Python path.
# This allows this script to find and import the SentimentAgent from the 'agents' folder.
# We go up two levels ('..') from the current file's directory (services/sentiment_service).
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# --- 2. AGENT IMPORT ---
# Now that the path is corrected, we can import the agent.
from agents.sentiment_agent import SentimentAgent

# --- 3. LOGGING & APP INITIALIZATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis Service",
    description="An API service to analyze the sentiment of financial news articles using FinBERT.",
    version="1.0.0"
)

# --- 4. PYDANTIC MODELS (for Data Validation) ---
# These models define the expected structure of your API's input and output.
# FastAPI uses them to validate incoming data and serialize outgoing data.

class Article(BaseModel):
    headline: Optional[str] = None
    content: Optional[str] = ""

class SentimentRequest(BaseModel):
    articles: List[Article] = Field(..., description="A list of articles to be analyzed.")

class SentimentResponse(BaseModel):
    sentiment_label: str
    sentiment_score: float
    note: Optional[str] = None

# --- 5. LOAD THE MODEL ONCE AT STARTUP ---
# We create the agent instance here, outside of the API endpoint function.
# This ensures the heavy FinBERT model is loaded into memory only once when the server starts,
# not every time a request is made, which is critical for performance.
try:
    sentiment_agent = SentimentAgent()
    logger.info("SentimentAgent loaded successfully on startup.")
except Exception as e:
    logger.error(f"Failed to load SentimentAgent on startup: {e}")
    sentiment_agent = None

# --- 6. API ENDPOINT ---
@app.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    """
    Accepts a list of articles and returns the aggregated sentiment analysis.
    """
    if not sentiment_agent:
        return {"sentiment_label": "Error", "sentiment_score": 0.0, "note": "Sentiment model is not available."}
    
    # The agent's run method expects a dictionary with an "articles" key.
    # We convert our Pydantic request model into this dictionary format.
    news_data = {"articles": [article.dict() for article in request.articles]}
    
    result = sentiment_agent.run(news_data)
    
    return result

# --- 7. (Optional) A root endpoint for health checks ---
@app.get("/")
def read_root():
    return {"status": "Sentiment Analysis Service is running."}