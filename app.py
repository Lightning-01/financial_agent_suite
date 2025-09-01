import streamlit as st
import pandas as pd
import os
import logging
from dotenv import load_dotenv
import requests

# Import all your agent classes
from agents.data_retriever_agent import DataRetrieverAgent
from agents.news_retriever_agent import NewsRetrieverAgent
from agents.summarizer_agent import SummarizerAgent
from agents.charting_agent import ChartingAgent
from agents.rule_based_engine import RuleBasedEngine
from agents.interpretation_agent import InterpretationAgent

# --- 1. SETUP AND CONFIGURATION ---

# Load API keys from your .env file
load_dotenv()

# --- TEMPORARY DEBUGGING ---
# print("--- Checking Loaded API Keys ---")
# print(f"NewsAPI Key: {os.getenv('NEWSAPI_KEY')}")
# print(f"GNews Key: {os.getenv('GNEWS_API_KEY')}")
# print(f"Alpha Vantage Key: {os.getenv('ALPHA_VANTAGE_API_KEY')}")
# print("----------------------------")

# Configure logging for the entire application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure the Streamlit page
st.set_page_config(layout="wide", page_title="AI Financial Analyst Suite")

# --- 2. AGENT INITIALIZATION (with Caching) ---

# Use Streamlit's caching to load heavy models only once
@st.cache_resource
def load_ai_agents():
    summarizer_agent = SummarizerAgent()
    return summarizer_agent

summarizer_agent = load_ai_agents()

# Instantiate the other, lighter agents
data_agent = DataRetrieverAgent()
news_agent = NewsRetrieverAgent()
chart_agent = ChartingAgent()
rule_engine = RuleBasedEngine()
interpretation_agent = InterpretationAgent()

# --- 3. USER INTERFACE ---

st.title("🤖 AI Financial Analyst Suite")

# Get user input for the stock ticker
ticker_input = st.text_input("Enter a stock ticker to analyze (e.g., AAPL, MSFT, NVDA):", "AAPL").upper()

if st.button("Generate Report"):
    if not ticker_input:
        st.error("Please enter a stock ticker.")
    else:
        with st.status("Kicking off the analysis...", expanded=True) as status:
            # Step 1: Data Retrieval
            status.update(label="Fetching financial data...", state="running")
            financial_data = data_agent.run(ticker_input)
            if financial_data.get("error"):
                status.update(label=f"Error: {financial_data['error']}", state="error", expanded=False)
                st.stop()
            status.update(label="Financial data retrieved!", state="complete")

            # Step 2: News Retrieval
            status.update(label="Gathering latest news...", state="running")
            company_name = financial_data.get('longName', ticker_input)
            short_name = financial_data.get('shortName') # Get the shortName
            # Pass all three identifiers to the agent
            news_data = news_agent.run(company_name, short_name, ticker_input)
            status.update(label="News gathered!", state="complete")

            # Step 3: AI Analysis
            status.update(label="Analyzing sentiment...", state="running")
            SENTIMENT_API_URL = "http://sentiment_service:8000/analyze"
            sentiment_result = {}
            try:
                # The news_data dictionary is already in the format {"articles": [...]} that our API expects
                response = requests.post(SENTIMENT_API_URL, json=news_data, timeout=30)
                response.raise_for_status()  # This will raise an error for bad responses (like 404 or 500)
                sentiment_result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to Sentiment Analysis service: {e}")
                # Provide a fallback error message so the app doesn't crash
                sentiment_result = {"sentiment_label": "Error", "sentiment_score": 0.0, "note": "Could not connect to the analysis service."}

            # The summarizer is still running locally for now
            status.update(label="Generating summary...", state="running")
            summary_result = summarizer_agent.run(news_data)
            status.update(label="AI analysis complete!", state="complete")

            # Step 4: Charting
            status.update(label="Creating visualizations...", state="running")
            chart_results = chart_agent.run(ticker_input)
            status.update(label="Charts are ready!", state="complete")

            # Step 5: Interpretation
            status.update(label="Interpreting technical data...", state="running")
            price_history_df = chart_results.get("price_history_df")
            interpretation_result = interpretation_agent.run(financial_data, price_history_df)
            status.update(label="Interpretation complete!", state="complete")

            # Step 6: Final Verdict
            status.update(label="Compiling final verdict...", state="running")
            verdict_result = rule_engine.run(financial_data, sentiment_result)
            status.update(label="Analysis Complete!", state="complete", expanded=False)


        # --- 5. DISPLAY THE FINAL REPORT ---
        
        # --- Header and Verdict ---
        # Create a fallback chain for the name: longName -> shortName -> Ticker
        display_name = financial_data.get('longName') or financial_data.get('shortName') or ticker_input
        st.header(f"Financial Report for {display_name}")
        st.subheader(f"Sector: {financial_data.get('sector')} | Industry: {financial_data.get('industry')}")
        
        st.header("Overall Analytical Tilt")

        # Get the results from the engine
        tilt = verdict_result.get('tilt', 'N/A')
        signal_strength = verdict_result.get('signal_strength', 'N/A')
        reason = verdict_result.get('reason', 'N/A')

        # Define color based on the tilt
        color = "green" if tilt == "Bullish" else "red" if tilt == "Bearish" else "orange"

        # Use a single, custom-styled markdown block for a professional look
        st.markdown(f"""
        <div style = 'padding-bottom: 16px;'>
            <h3 style='color:{color}; margin-top: 0px;'>{tilt}</h3>
            <p style="font-size: 1.1em; margin-bottom: 8px;"><strong>Signal Strength:</strong> {signal_strength}</p>
            <p style="font-size: 1em; margin-bottom: 0px;"><strong>Thesis:</strong> {reason}</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Charts ---
        st.header("Charts & Visualizations")
        price_chart = chart_results.get("price_chart")
        financials_chart = chart_results.get("financials_chart")
        
        if price_chart and financials_chart:
            st.plotly_chart(price_chart, use_container_width=True)
            st.plotly_chart(financials_chart, use_container_width=True)
        else:
            st.warning("One or more charts could not be displayed.")

        st.header("Technical Analysis Summary") 
        for insight in interpretation_result.get('technical_insights', []):
            st.markdown(f"- {insight}")

        # --- Scorecard and News Summary ---
        with st.expander("Show Detailed Analytical Scorecard"):
            st.table(pd.DataFrame(verdict_result.get('scorecard', [])))
        
        with st.expander("Show Recent News Summary"):
            st.write(summary_result.get('summary', 'No summary available.'))
            st.markdown(f"**Overall News Sentiment:** {sentiment_result.get('sentiment_label')} (Score: {sentiment_result.get('sentiment_score')})")

        # --- Raw Data ---
        with st.expander("Show Raw Financial Data"):
            st.json(financial_data)