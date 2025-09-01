import logging
import pandas as pd
import numpy as np
# from agents.data_retriever_agent import DataRetrieverAgent # For testing
# from agents.charting_agent import ChartingAgent # For testing

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InterpretationAgent:
    """
    Analyzes historical price data to generate a summary of key technical events and trends.
    """
    def run(self, financial_data: dict, price_history_df: pd.DataFrame) -> dict:
        """
        Main entry point for the agent.
        
        Args:
            financial_data: The dictionary output from the DataRetrieverAgent.
            price_history_df: The pandas DataFrame of historical prices from yfinance.

        Returns:
            A dictionary containing a list of technical insight strings.
        """
        logger.info(f"InterpretationAgent: Analyzing technicals for {financial_data.get('ticker', 'N/A')}...")
        
        # Check for valid inputs
        if price_history_df is None or price_history_df.empty:
            logger.warning("No price history data provided. Skipping interpretation.")
            return {"technical_insights": ["No historical data available for analysis."]}
        
        current_price = financial_data.get('price')
        if not current_price:
            # Fallback to the last known close price if real-time price is missing
            current_price = price_history_df['Close'].iloc[-1]

        insights = []
        try:
            # Generate insights from each analysis function
            insights.extend(self._analyze_performance(price_history_df, current_price))
            insights.extend(self._analyze_moving_averages(price_history_df, current_price))
            insights.extend(self._analyze_volatility(price_history_df))
            
            # This check is last as it's an optional/rare event
            crossover_insight = self._find_ma_crossover(price_history_df)
            if crossover_insight:
                insights.append(crossover_insight)

            logger.info(f"Successfully generated {len(insights)} technical insights.")
            return {"technical_insights": insights}
        except Exception as e:
            logger.error(f"InterpretationAgent: An error occurred during analysis: {e}")
            return {"technical_insights": ["An error occurred during technical analysis."], "error": str(e)}

    def _analyze_performance(self, df: pd.DataFrame, current_price: float) -> list[str]:
        """Calculates 1-year return and position vs. 52-week range."""
        insights = []
        price_one_year_ago = df['Close'].iloc[0]
        year_return = ((current_price - price_one_year_ago) / price_one_year_ago) * 100
        insights.append(f"The stock has returned {year_return:.2f}% over the last year.")
        
        high_52_week = df['Close'].max()
        low_52_week = df['Close'].min()
        percent_below_high = ((high_52_week - current_price) / high_52_week) * 100
        insights.append(f"It is currently trading {percent_below_high:.2f}% below its 52-week high of ${high_52_week:.2f}.")
        
        return insights

    def _analyze_moving_averages(self, df: pd.DataFrame, current_price: float) -> list[str]:
        """Compares the current price to key moving averages."""
        insights = []
        sma50 = df['Close'].rolling(window=50).mean().iloc[-1]
        sma200 = df['Close'].rolling(window=200).mean().iloc[-1]

        if current_price > sma50:
            insights.append("The current price is above its 50-day moving average, indicating positive short-term momentum.")
        else:
            insights.append("The current price is below its 50-day moving average, indicating negative short-term momentum.")
        
        if current_price > sma200:
            insights.append("The price is above its 200-day moving average, suggesting a positive long-term trend.")
        else:
            insights.append("The price is below its 200-day moving average, suggesting a negative long-term trend.")
            
        return insights

    def _analyze_volatility(self, df: pd.DataFrame) -> list[str]:
        """Calculates the annualized historical volatility."""
        daily_returns = df['Close'].pct_change()
        # Annualized volatility = daily standard deviation * sqrt(252 trading days)
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        return [f"The stock has an annualized volatility of {annualized_volatility:.2%}, indicating its level of price fluctuation."]

    def _find_ma_crossover(self, df: pd.DataFrame) -> str | None:
        """Finds the most recent Golden Cross or Death Cross event."""
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Create a column to represent the position: 1 if SMA50 > SMA200, else -1
        df['Position'] = np.where(df['SMA50'] > df['SMA200'], 1, -1)
        # Find the difference from the previous day to identify crossover points
        df['Crossover'] = df['Position'].diff()
        
        # Find the last crossover event
        last_crossover = df[df['Crossover'] != 0].tail(1)
        if not last_crossover.empty:
            crossover_date = last_crossover.index[0].strftime('%Y-%m-%d')
            crossover_type = last_crossover['Crossover'].iloc[0]
            if crossover_type == 2.0: # Flipped from -1 to 1
                return f"A 'Golden Cross' (50-day MA crossing above 200-day MA) occurred on {crossover_date}, a bullish signal."
            elif crossover_type == -2.0: # Flipped from 1 to -1
                return f"A 'Death Cross' (50-day MA crossing below 200-day MA) occurred on {crossover_date}, a bearish signal."
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # To test this agent, we need data from the other agents first.
    # This block simulates the data flow.
    import sys
    import os
    # Add the parent directory (your main project folder) to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # --- END OF FIX ---

    # Now the original imports will work correctly for testing
    from agents.data_retriever_agent import DataRetrieverAgent
    from agents.charting_agent import ChartingAgent
    import yfinance as yf
    data_agent = DataRetrieverAgent()
    interpretation_agent = InterpretationAgent()
    
    ticker = "MSFT"
    
    # 1. Get financial data (for current price)
    financial_data = data_agent.run(ticker)
    
    # 2. Get historical price data (this would normally come from the charting agent)
    stock = yf.Ticker(ticker)
    price_history = stock.history(period="1y")

    # 3. Run the interpretation agent
    if financial_data and not price_history.empty:
        insights_result = interpretation_agent.run(financial_data, price_history)
        print(f"\n--- Technical Insights for {ticker} ---")
        for insight in insights_result.get("technical_insights", []):
            print(f"- {insight}")