import logging
import os
import requests

logger = logging.getLogger(__name__)
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

class SentimentAgent:
    def run(self, news_data: dict) -> dict:
        articles = news_data.get("articles", [])
        if not articles:
            return {"sentiment_label": "Neutral", "sentiment_score": 0.0, "note": "No articles for analysis."}
        
        texts_to_analyze = [a.get('content') or a.get('headline', '') for a in articles]
        logger.info(f"SentimentAgent: Analyzing {len(texts_to_analyze)} articles via Inference API...")
        
        try:
            response = requests.post(API_URL, headers=HEADERS, json={"inputs": texts_to_analyze})
            response.raise_for_status()
            sentiments = response.json()
            
            score_sum = 0
            for label_scores in sentiments:
                scores = {d['label']: d['score'] for d in label_scores}
                score_sum += scores.get('positive', 0.0) - scores.get('negative', 0.0)

            average_score = score_sum / len(sentiments) if sentiments else 0.0
            
            if average_score > 0.1: final_label = "Positive"
            elif average_score < -0.1: final_label = "Negative"
            else: final_label = "Neutral"
            
            return {"sentiment_label": final_label, "sentiment_score": round(average_score, 4)}
        except Exception as e:
            logger.error(f"SentimentAgent API Error: {e}")
            return {"sentiment_label": "Error", "sentiment_score": 0.0, "note": str(e)}