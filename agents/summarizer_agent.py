import logging
import os
import requests

logger = logging.getLogger(__name__)
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
# API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
API_URL = "https://router.huggingface.co/hf-inference/models/sshleifer/distilbart-cnn-12-6"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

class SummarizerAgent:
    def run(self, news_data: dict) -> dict:
        articles = news_data.get("articles", [])
        if not articles:
            return {"summary": "No news content available to generate a summary."}

        # texts_to_summarize = [
        #     (a.get('headline', '') or "") + ". " + (a.get('content', '') or "")
        #     for a in articles if a.get('content') or a.get('headline')
        # ]
        # if not texts_to_summarize:
        #     return {"summary": "No article content available to generate a summary."}

        # 1. Take only the top 5 articles to keep the payload manageable
        top_articles = articles[:5]

        texts_to_summarize = []
        for a in top_articles:
            headline = a.get('headline', '') or ""
            content = a.get('content', '') or ""
            # Combine headline + content
            full_text = f"{headline}. {content}"
            # 2. Truncate each article to 600 chars so we don't overflow the model
            texts_to_summarize.append(full_text[:600])

        if not texts_to_summarize:
            return {"summary": "No article content available to generate a summary."}

        combined_text = " ".join(texts_to_summarize)

        # 3. Final safety check: Limit total to ~3000 chars
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000]

        logger.info(f"SummarizerAgent: Summarizing via Inference API...")

        try:
            payload = {
                "inputs": combined_text,
                "parameters": {"min_length": 50, "max_length": 250}
            }
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            # summary = response.json()[0]['summary_text']
            # return {"summary": summary}
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                summary = result[0].get('summary_text', "No summary text returned.")
            else:
                summary = "Unexpected API response format (possibly model loading)."

            return {"summary": summary}
        except Exception as e:
            logger.error(f"SummarizerAgent API Error: {e}")
            return {"summary": "Error generating summary.", "error": str(e)}