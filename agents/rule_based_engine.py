import logging

logger = logging.getLogger(__name__)

class RuleBasedEngine:
    """
    Analyzes data to generate an investment recommendation based on a "Quality & Growth" philosophy.
    """
    WEIGHTS = {"Debt-to-Equity": 1.5, "Profit Margin": 1.2, "Return on Equity": 1.0, "News Sentiment": 1.0, "P/E Ratio": 0.8, "Price-to-Sales Ratio": 0.8}
    BUY_THRESHOLD = 1.5
    SELL_THRESHOLD = -1.5

    def run(self, financial_data: dict, sentiment_data: dict) -> dict:
        logger.info(f"RuleBasedEngine: Analyzing {financial_data.get('ticker', 'N/A')}...")
        scorecard = [
            self._check_pe_ratio(financial_data.get('pe_ratio')),
            self._check_ps_ratio(financial_data.get('price_to_sales')),
            self._check_profit_margin(financial_data.get('profit_margins')),
            self._check_roe(financial_data.get('return_on_equity')),
            self._check_debt_to_equity(financial_data.get('debt_to_equity')),
            self._check_sentiment(sentiment_data.get('sentiment_score'))
        ]
        return self._calculate_final_verdict(scorecard)

    def _calculate_final_verdict(self, scorecard: list) -> dict:
        total_score = 0
        positive_checks = 0
        negative_checks = 0
        rating_map = {"Positive": 1, "Neutral": 0, "Negative": -1}

        for check in scorecard:
            weight = self.WEIGHTS.get(check['metric'], 1.0)
            total_score += rating_map[check['rating']] * weight
            if check['rating'] == "Positive":
                positive_checks += 1
            elif check['rating'] == "Negative":
                negative_checks += 1

        # Determine the Tilt based on the weighted score
        if total_score >= self.BUY_THRESHOLD:
            tilt = "Bullish"
        elif total_score <= self.SELL_THRESHOLD:
            tilt = "Bearish"
        else:
            tilt = "Neutral"

        # Create a clear, unambiguous Signal Strength and Reason
        signal_strength = f"{positive_checks} Positive | {negative_checks} Negative"
        reason = (f"The analysis found {positive_checks} positive and {negative_checks} negative factors "
                f"out of {len(scorecard)} total checks, resulting in a weighted score of {total_score:.2f}.")

        return {
            "tilt": tilt,
            "signal_strength": signal_strength,
            "reason": reason,
            "scorecard": scorecard
        }
    
    def _check_pe_ratio(self, value):
        rating = "Neutral"
        if isinstance(value, (int, float)):
            if value < 20: rating = "Positive"
            elif value > 40: rating = "Negative"
        return {"metric": "P/E Ratio", "value": value, "benchmark": "20 < P/E < 40", "rating": rating}

    def _check_ps_ratio(self, value):
        rating = "Neutral"
        if isinstance(value, (int, float)):
            if value < 2: rating = "Positive"
            elif value > 5: rating = "Negative"
        return {"metric": "Price-to-Sales Ratio", "value": value, "benchmark": "P/S < 5", "rating": rating}

    def _check_profit_margin(self, value):
        rating = "Neutral"
        if isinstance(value, (int, float)):
            if value > 0.20: rating = "Positive"
            elif value < 0.10: rating = "Negative"
        return {"metric": "Profit Margin", "value": value, "benchmark": "> 10%", "rating": rating}

    def _check_roe(self, value):
        rating = "Neutral"
        if isinstance(value, (int, float)):
            if value > 0.15: rating = "Positive"
            elif value < 0.05: rating = "Negative"
        return {"metric": "Return on Equity", "value": value, "benchmark": "> 5%", "rating": rating}

    def _check_debt_to_equity(self, value):
        rating = "Neutral"
        if isinstance(value, (int, float)):
            if value < 1.0: rating = "Positive"
            elif value > 2.0: rating = "Negative"
        return {"metric": "Debt-to-Equity", "value": value, "benchmark": "< 2.0", "rating": rating}

    def _check_sentiment(self, value):
        rating = "Neutral"
        if isinstance(value, (int, float)):
            if value > 0.15: rating = "Positive"
            elif value < -0.15: rating = "Negative"
        return {"metric": "News Sentiment", "value": value, "benchmark": "> -0.15", "rating": rating}