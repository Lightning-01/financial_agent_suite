import logging
from functools import lru_cache
from datetime import datetime
import yfinance as yf

logger = logging.getLogger(__name__)

class DataRetrieverAgent:
    """
    Retrieves fundamental and profile data for a stock using yfinance.
    Caches results to avoid redundant network requests.
    """
    @lru_cache(maxsize=128)
    def run(self, ticker_symbol: str) -> dict:
        logger.info(f"DataRetrieverAgent: Fetching data for {ticker_symbol.upper()}...")
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            if not info or 'symbol' not in info:
                raise ValueError(f"No data for ticker '{ticker_symbol}'. It may be invalid.")
            
            fundamental_data = self._get_fundamental_metrics(info)
            profile_data = self._get_profile_metrics(info)
            
            return {
                "ticker": ticker_symbol.upper(),
                **fundamental_data,
                **profile_data,
                "error": None
            }
        except Exception as e:
            logger.error(f"DataRetrieverAgent: Error for {ticker_symbol.upper()}. Reason: {e}")
            return {"ticker": ticker_symbol.upper(), "error": str(e)}

    def _get_fundamental_metrics(self, info: dict) -> dict:
        return {
            'pe_ratio': info.get('trailingPE'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'price_to_book': info.get('priceToBook'),
            'profit_margins': info.get('profitMargins'),
            'return_on_equity': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'beta': info.get('beta'),
            'forward_dividend_yield': info.get('forwardDividendYield'),
        }

    def _get_profile_metrics(self, info: dict) -> dict:
        next_earnings_date = None
        earnings_dates = info.get("earningsDate")
        if isinstance(earnings_dates, list) and earnings_dates and isinstance(earnings_dates[0], datetime):
            next_earnings_date = earnings_dates[0].strftime("%Y-%m-%d")
        return {
            'shortName': info.get('shortName'), 
            'longName': info.get('longName'),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'long_business_summary': info.get('longBusinessSummary'),
            'next_earnings_date': next_earnings_date,
        }