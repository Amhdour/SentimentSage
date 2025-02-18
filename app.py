import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import json
from utils import preprocess_text, calculate_reputation_score
from sample_data import get_sample_data
from openai_analyzer import analyze_sentiment_openai

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Sentiment Analysis & Reputation Management Dashboard")
st.markdown("""
    This dashboard provides real-time sentiment analysis and reputation management insights 
    from text data. Upload your own data or explore our sample dataset.
""")

# Sidebar
st.sidebar.header("Dashboard Controls")
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Sample Data", "Upload Your Own"]
)

# Data loading
if data_source == "Sample Data":
    df = get_sample_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file or select Sample Data")
        st.stop()

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("Overall Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    fig_sentiment = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

with col2:
    st.subheader("Sentiment Score Distribution")
    fig_scores = px.histogram(
        df,
        x="polarity",
        nbins=20,
        title="Distribution of Sentiment Scores",
        color_discrete_sequence=['#1f77b4']
    )
    fig_scores.update_layout(showlegend=False)
    st.plotly_chart(fig_scores, use_container_width=True)

# Real-time Analysis Section
st.header("🔄 Real-time Sentiment Analysis")
st.markdown("""
    Enter text below to analyze sentiment in real-time using both TextBlob and OpenAI's advanced AI model.
    Compare the results to see how different models interpret the sentiment.
""")

# Text input for real-time analysis
user_text = st.text_area("Enter text to analyze:", height=150)

if user_text:
    # Create three columns for analysis results
    col1, col2 = st.columns(2)

    # TextBlob Analysis
    with col1:
        st.subheader("TextBlob Analysis")
        processed_text = preprocess_text(user_text)
        blob = TextBlob(processed_text)

        st.metric("Polarity", f"{blob.sentiment.polarity:.2f}")
        st.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")
        sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"
        st.metric("Sentiment", sentiment)

    # OpenAI Analysis
    with col2:
        st.subheader("OpenAI Advanced Analysis")
        with st.spinner("Analyzing sentiment with AI..."):
            openai_result = json.loads(analyze_sentiment_openai(user_text))

            st.metric("Sentiment", openai_result['sentiment'])
            st.metric("Confidence", f"{openai_result['confidence']:.2f}")
            st.metric("Emotional Tone", openai_result['emotional_tone'])

            st.write("Key Sentiment Drivers:")
            for driver in openai_result['key_drivers']:
                st.write(f"• {driver}")

# Reputation Score
st.subheader("Reputation Score")
reputation_score = calculate_reputation_score(df)
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=reputation_score,
    title={'text': "Overall Reputation Score"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#1f77b4"},
        'steps': [
            {'range': [0, 33], 'color': "lightgray"},
            {'range': [33, 66], 'color': "gray"},
            {'range': [66, 100], 'color': "darkgray"}
        ]
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)


# Historical Data Table
st.header("📊 Historical Data")
st.dataframe(df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit, TextBlob, and OpenAI")