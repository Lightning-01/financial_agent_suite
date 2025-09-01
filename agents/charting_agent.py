import logging
from functools import lru_cache
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

logger = logging.getLogger(__name__)

class ChartingAgent:
    """
    Creates interactive financial charts using Plotly.
    """
    @lru_cache(maxsize=128)
    def run(self, ticker_symbol: str) -> dict:
        logger.info(f"ChartingAgent: Generating charts for {ticker_symbol.upper()}...")
        try:
            stock = yf.Ticker(ticker_symbol)
            # Fetch history once to be used by both chart and interpretation agents
            price_history_df = stock.history(period="1y")
            price_fig = self._create_price_volume_chart(stock, price_history_df)
            financials_fig = self._create_financials_chart(stock)
            if not price_fig and not financials_fig: raise ValueError("Could not generate any charts.")
            return {
                "price_chart": price_fig,
                "financials_chart": financials_fig,
                "price_history_df": price_history_df,
                "error": None
            }
        except Exception as e:
            logger.error(f"ChartingAgent: Failed for {ticker_symbol.upper()}. Reason: {e}")
            return {"error": str(e)}

    def _create_price_volume_chart(self, stock: yf.Ticker, hist: pd.DataFrame) -> go.Figure | None:
        try:
            if hist.empty: return None
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(window=50).mean(), name='50-Day MA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(window=200).mean(), name='200-Day MA'), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume'), row=2, col=1)
            fig.update_layout(title_text=f"{stock.ticker} Price History (1Y)", xaxis_rangeslider_visible=False, template="plotly_white")
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            return fig
        except Exception as e:
            logger.error(f"Could not create price/volume chart for {stock.ticker}: {e}")
            return None

    def _create_financials_chart(self, stock: yf.Ticker) -> go.Figure | None:
        try:
            financials = stock.quarterly_financials
            if financials.empty: return None
            df = financials.T[['Total Revenue', 'Net Income']].sort_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df.index.strftime('%Y-%m-%d'), y=df['Total Revenue'], name='Revenue'))
            fig.add_trace(go.Bar(x=df.index.strftime('%Y-%m-%d'), y=df['Net Income'], name='Net Income'))
            fig.update_layout(title_text=f"{stock.ticker} Quarterly Revenue & Income", yaxis_title="Amount (USD)", barmode='group', template="plotly_white")
            return fig
        except Exception as e:
            logger.error(f"Could not create financials chart for {stock.ticker}: {e}")
            return None
