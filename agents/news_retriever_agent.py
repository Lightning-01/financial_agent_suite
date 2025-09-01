import logging
import os
import requests
from functools import lru_cache
from datetime import datetime, timedelta, UTC
from thefuzz import fuzz

logger = logging.getLogger(__name__)
# These lines will read the keys that are loaded by the main app.py
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

class NewsRetrieverAgent:
    """
    Aggregates news from multiple sources with a dynamic time window and fuzzy deduplication.
    """
    @lru_cache(maxsize=128)
    def run(self, company_name: str, short_name: str, stock_ticker: str) -> dict:
        logger.info(f"NewsRetrieverAgent: Fetching news for {company_name}...")
        search_windows = [7, 30]
        min_articles_threshold = 10
        all_articles = []
        for days_back in search_windows:
            logger.info(f"...searching within the last {days_back} days.")
            # Pass the stock_ticker to the fetch methods
            fetched = (self._fetch_from_newsapi(company_name, stock_ticker, days_back) +
                       self._fetch_from_gnews(company_name, stock_ticker, days_back) +
                       self._fetch_from_alpha_vantage(stock_ticker, days_back))
            all_articles.extend(fetched)
            if len(all_articles) >= min_articles_threshold:
                logger.info(f"Found sufficient articles within {days_back} days.")
                break
        
        unique_articles = self._deduplicate_articles(all_articles)
        
        relevant_articles = []
    
        # Use the clean short_name for filtering if it exists, otherwise it's None
        name_to_check = short_name.lower() if short_name else None

        for article in unique_articles:
            headline = article.get('headline', '').lower()
            
            # An article is relevant if the ticker is in the headline.
            # If we also have a clean name, it's even better if that's also in the headline.
            # But we prioritize just having one of them to not filter too much.
            
            is_relevant = False
            if stock_ticker.lower() in headline:
                is_relevant = True
            
            if name_to_check and name_to_check in headline:
                is_relevant = True

            if is_relevant:
                relevant_articles.append(article)

        if not relevant_articles and unique_articles:
            logger.warning(f"No highly relevant news found for {company_name}. Showing broader results.")
            final_articles = unique_articles[:20]
        else:
            final_articles = relevant_articles[:20]

        logger.info(f"NewsRetrieverAgent: Found {len(unique_articles)} articles, returning {len(final_articles)} after relevancy filtering.")

        return {"articles": final_articles}

    def _deduplicate_articles(self, articles: list) -> list:
        unique_articles = []
        seen_headlines = []
        similarity_threshold = 85
        for article in articles:
            if not article.get('headline'): continue
            is_duplicate = any(fuzz.token_set_ratio(article['headline'], seen) > similarity_threshold for seen in seen_headlines)
            if not is_duplicate:
                unique_articles.append(article)
                seen_headlines.append(article['headline'])
        return unique_articles

    def _calculate_date_range(self, days_back: int) -> tuple[datetime, datetime]:
        to_date = datetime.now(UTC)
        from_date = to_date - timedelta(days=days_back)
        return from_date, to_date

    def _build_search_query(self, company_name: str, ticker: str) -> str:
        """Constructs a robust search query to avoid ambiguity."""
        # If we have a proper company name that isn't just the ticker, prioritize it.
        if company_name and company_name.lower() != ticker.lower():
            return f'"{company_name}" OR "{ticker} stock"'
        # Otherwise, use the ticker but add context words.
        else:
            return f'"{ticker} stock" OR "{ticker} company"'

    def _fetch_from_newsapi(self, query: str, ticker: str, days_back: int) -> list:
        logger.info("...fetching from NewsAPI")
        search_query = self._build_search_query(query, ticker)
        from_date, to_date = self._calculate_date_range(days_back)
        try:
            url = (f"https://newsapi.org/v2/everything?q={search_query}&apiKey={NEWSAPI_KEY}"
                   f"&from={from_date.isoformat()}&to={to_date.isoformat()}"
                   f"&pageSize=20&sortBy=relevancy&language=en")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return [{"source": "NewsAPI", "headline": a.get("title"), "url": a.get("url"), "content": a.get("description") or ""} for a in response.json().get("articles", [])]
        except Exception as e:
            logger.error(f"Failed to fetch from NewsAPI: {e}")
            return []

    def _fetch_from_gnews(self, query: str, ticker: str, days_back: int) -> list:
        logger.info("...fetching from GNews")
        search_query = self._build_search_query(query, ticker)
        from_date, _ = self._calculate_date_range(days_back)
        try:
            from_iso = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            # GNews requires quotes for exact phrases within the query itself
            url = f"https://gnews.io/api/v4/search?q={search_query}&apikey={GNEWS_API_KEY}&max=20&lang=en&from={from_iso}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return [{"source": "GNews", "headline": a.get("title"), "url": a.get("url"), "content": a.get("description") or ""} for a in response.json().get("articles", [])]
        except Exception as e:
            logger.error(f"Failed to fetch from GNews: {e}")
            return []
            
    def _fetch_from_alpha_vantage(self, ticker: str, days_back: int) -> list:
        logger.info("...fetching from Alpha Vantage")
        from_date, _ = self._calculate_date_range(days_back)
        try:
            time_from = from_date.strftime('%Y%m%dT%H%M')
            url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
                   f"&tickers={ticker}&apikey={ALPHA_VANTAGE_API_KEY}&limit=20&time_from={time_from}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            articles = data.get("feed", [])

            # Enforce the limit on our side, as the API's limit parameter is not always reliable.
            articles = articles[:20]

            return [{"source": "Alpha Vantage", "headline": a.get("title"), "url": a.get("url"), "content": a.get("summary") or ""} for a in articles]
        except Exception as e:
            logger.error(f"Failed to fetch from Alpha Vantage: {e}")
            return []