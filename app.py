import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from utils import preprocess_text, calculate_reputation_score
from sample_data import get_sample_data
from openai_analyzer import analyze_sentiment_openai
from nlp_analyzer import get_nlp_analysis

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
    from text data, including advanced NLP features like entity recognition and topic modeling.
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
st.header("🔄 Real-time Text Analysis")
st.markdown("""
    Enter text below for comprehensive analysis including:
    - Basic sentiment analysis using TextBlob
    - Advanced sentiment analysis with OpenAI (when available)
    - Named Entity Recognition
    - Topic Modeling
    - Key Phrase Extraction
""")

# Text input for analysis
user_text = st.text_area("Enter text to analyze:", height=150)

if user_text:
    # TextBlob Analysis
    processed_text = preprocess_text(user_text)
    blob = TextBlob(processed_text)

    # Create tabs for different analyses
    sentiment_tab, entities_tab, topics_tab = st.tabs([
        "Sentiment Analysis",
        "Named Entities",
        "Topics & Phrases"
    ])

    # Sentiment Analysis Tab
    with sentiment_tab:
        try:
            openai_result = analyze_sentiment_openai(user_text)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("TextBlob Analysis")
                st.metric("Polarity", f"{blob.sentiment.polarity:.2f}")
                st.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")
                sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"
                st.metric("Sentiment", sentiment)

            with col2:
                st.subheader("OpenAI Advanced Analysis")
                st.metric("Sentiment", openai_result['sentiment'])
                st.metric("Confidence", f"{openai_result['confidence']:.2f}")
                st.metric("Emotional Tone", openai_result['emotional_tone'])

                st.write("Key Sentiment Drivers:")
                for driver in openai_result['key_drivers']:
                    st.write(f"• {driver}")

        except Exception as e:
            st.warning(f"Advanced AI analysis unavailable: {str(e)}")
            st.subheader("TextBlob Analysis")
            st.metric("Polarity", f"{blob.sentiment.polarity:.2f}")
            st.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")
            sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"
            st.metric("Sentiment", sentiment)

    # Get advanced NLP analysis
    nlp_results = get_nlp_analysis(user_text)

    # Entities Tab
    with entities_tab:
        st.subheader("Named Entity Recognition")

        if nlp_results['entities']:
            for entity_type, entities in nlp_results['entities'].items():
                st.write(f"**{entity_type}**")
                for entity in entities:
                    st.write(f"• {entity['text']} (mentioned {entity['count']} times)")
        else:
            st.info("No named entities found in the text.")

    # Topics and Phrases Tab
    with topics_tab:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Key Phrases")
            phrases = nlp_results.get('key_phrases', [])
            if phrases:
                for phrase in phrases:
                    st.write(f"• {phrase['text']} ({phrase['type']})")
            else:
                st.info("No key phrases identified.")

        with col2:
            st.subheader("Topic Analysis")
            if 'topics' in nlp_results:
                for topic in nlp_results['topics']:
                    st.write(f"**Topic {topic['id'] + 1}**")
                    for word_info in topic['words']:
                        st.write(f"• {word_info['word']} ({word_info['weight']:.3f})")
            else:
                st.info("Text is too short for topic modeling. Please provide a longer text.")

# Reputation Score
st.header("📈 Reputation Analysis")
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
st.markdown("Dashboard created with Streamlit, TextBlob, OpenAI, and spaCy")