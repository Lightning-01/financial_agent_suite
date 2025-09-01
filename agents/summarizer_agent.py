import logging
import torch
from transformers.pipelines import pipeline

logger = logging.getLogger(__name__)

class SummarizerAgent:
    """
    Uses DistilBART to summarize news articles with a map-reduce strategy.
    """
    def __init__(self):
        logger.info("SummarizerAgent: Initializing and loading DistilBART model...")
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
        logger.info("SummarizerAgent: Model loaded successfully.")

    def run(self, news_data: dict) -> dict:
        articles = news_data.get("articles", [])
        if not articles: return {"summary": "No news content available to generate a summary."}

        texts_to_summarize = [
            (a.get('headline', '') or "") + ". " + (a.get('content', '') or "")
            for a in articles if a.get('content') or a.get('headline')
        ]
        if not texts_to_summarize:
            return {"summary": "No article content available to generate a summary."}


        # 1. Combine all text snippets into a single block of text
        combined_text = " ".join(texts_to_summarize)
        
        logger.info(f"SummarizerAgent: Starting single-pass summarization on combined text of {len(combined_text)} characters.")

        try:
            # 2. Perform a single summarization pass on the combined text
            final_summary_list = self.summarizer_pipeline(
                combined_text, max_length=250, min_length=50, truncation=True
            )
            return {"summary": final_summary_list[0]['summary_text']}
        except Exception as e:
            logger.error(f"SummarizerAgent: Error during summarization. Reason: {e}")
            return {"summary": "Error generating summary.", "error": str(e)}
