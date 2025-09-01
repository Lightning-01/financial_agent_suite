import logging
import os
import requests

logger = logging.getLogger(__name__)
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

class SummarizerAgent:
    def run(self, news_data: dict) -> dict:
        articles = news_data.get("articles", [])
        if not articles:
            return {"summary": "No news content available to generate a summary."}

        texts_to_summarize = [
            (a.get('headline', '') or "") + ". " + (a.get('content', '') or "")
            for a in articles if a.get('content') or a.get('headline')
        ]
        if not texts_to_summarize:
            return {"summary": "No article content available to generate a summary."}

        combined_text = " ".join(texts_to_summarize)
        logger.info(f"SummarizerAgent: Summarizing via Inference API...")

        try:
            payload = {
                "inputs": combined_text,
                "parameters": {"min_length": 50, "max_length": 250}
            }
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            summary = response.json()[0]['summary_text']
            return {"summary": summary}
        except Exception as e:
            logger.error(f"SummarizerAgent API Error: {e}")
            return {"summary": "Error generating summary.", "error": str(e)}