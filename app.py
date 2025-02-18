import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from utils import preprocess_text, calculate_reputation_score
from sample_data import get_sample_data

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Sentiment Analysis & Reputation Management Dashboard")
st.markdown("""
    This dashboard provides sentiment analysis and reputation management insights 
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

# Text Analysis Section
st.subheader("Live Text Analysis")
user_text = st.text_area("Enter text to analyze:")
if user_text:
    processed_text = preprocess_text(user_text)
    blob = TextBlob(processed_text)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Polarity", f"{blob.sentiment.polarity:.2f}")
    with col4:
        st.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")
    with col5:
        sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"
        st.metric("Sentiment", sentiment)

# Data Table
st.subheader("Recent Data")
st.dataframe(df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit and TextBlob")
