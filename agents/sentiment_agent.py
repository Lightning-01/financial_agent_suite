import logging
import torch
from transformers.pipelines import pipeline

logger = logging.getLogger(__name__)

class SentimentAgent:
    """
    Uses FinBERT to analyze the sentiment of a list of news articles.
    """
    def __init__(self):
        logger.info("SentimentAgent: Initializing and loading FinBERT model...")
        device = 0 if torch.cuda.is_available() else -1
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
        logger.info("SentimentAgent: FinBERT model loaded successfully.")

    def run(self, news_data: dict) -> dict:
        articles = news_data.get("articles", [])
        if not articles:
            return {"sentiment_label": "Neutral", "sentiment_score": 0.0, "note": "No articles for analysis."}
        
        texts_to_analyze = [a.get('content') or a.get('headline', '') for a in articles]
        logger.info(f"SentimentAgent: Analyzing {len(texts_to_analyze)} articles...")
        try:
            sentiments = self.sentiment_pipeline(texts_to_analyze)
            score_sum = sum((s['score'] if s['label'] == 'positive' else -s['score']) for s in sentiments if s['label'] != 'neutral')
            average_score = score_sum / len(sentiments)
            
            if average_score > 0.1: final_label = "Positive"
            elif average_score < -0.1: final_label = "Negative"
            else: final_label = "Neutral"
            
            return {"sentiment_label": final_label, "sentiment_score": round(average_score, 4)}
        except Exception as e:
            logger.error(f"SentimentAgent: Error during analysis. Reason: {e}")
            return {"sentiment_label": "Error", "sentiment_score": 0.0, "note": str(e)}