import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from utils import preprocess_text, calculate_reputation_score
from sample_data import get_sample_data
from openai_analyzer import analyze_sentiment_openai
from nlp_analyzer import get_nlp_analysis
from data_sources import load_data_source, load_from_urls
from ml_predictor import get_trend_analysis

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
    from various data sources, including files, JSON input, and web content.
""")

# Sidebar
st.sidebar.header("Dashboard Controls")
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Sample Data", "Upload File", "JSON Input", "Web URLs"]
)

# Data loading
if data_source == "Sample Data":
    df = get_sample_data()
elif data_source == "Web URLs":
    urls_input = st.sidebar.text_area(
        "Enter URLs (one per line)",
        height=150,
        help="Enter website URLs to analyze their content. Each URL should be on a new line."
    )

    if urls_input:
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        if urls:
            with st.spinner('Fetching content from URLs...'):
                df, error = load_from_urls(urls)
                if error:
                    st.sidebar.error(error)
                    st.stop()
        else:
            st.info("Please enter valid URLs or select another data source")
            st.stop()
    else:
        st.info("Please enter URLs or select another data source")
        st.stop()
elif data_source == "JSON Input":
    json_text = st.sidebar.text_area(
        "Enter JSON data",
        height=200,
        help="""Enter JSON data in the format:
        [{\"text\": \"your text here\"}, ...]
        or
        {\"data\": [{\"text\": \"your text here\"}, ...]}"""
    )
    if json_text:
        try:
            from io import StringIO
            json_file = StringIO(json_text)
            df, error = load_data_source(json_file, 'json')
            if error:
                st.sidebar.error(error)
                st.stop()
        except Exception as e:
            st.sidebar.error(f"Error parsing JSON: {str(e)}")
            st.stop()
    else:
        st.info("Please enter JSON data or select another data source")
        st.stop()
else:  # Upload File
    uploaded_file = st.sidebar.file_uploader(
        "Upload data file", 
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload a file containing text data for analysis. Must include a 'text' column."
    )

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        with st.spinner('Processing uploaded file...'):
            df, error = load_data_source(uploaded_file, file_type)

            if error:
                st.sidebar.error(error)
                st.stop()

            # Display data preview
            st.sidebar.subheader("Data Preview")
            st.sidebar.dataframe(df.head(3), use_container_width=True)
    else:
        st.info("Please upload a file or select another data source")
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

# Add Trend Analysis Section
st.header("📈 Trend Analysis")
if len(df) >= 10:  # Only show predictions if we have enough data
    with st.spinner("Analyzing sentiment trends..."):
        trend_results = get_trend_analysis(df)

        if trend_results['error']:
            st.error(trend_results['error'])
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Predicted Sentiment Trend")
                predictions = trend_results['predictions']

                # Plot historical and predicted sentiment
                fig_trend = go.Figure()

                # Historical data
                fig_trend.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['polarity'],
                    name='Historical',
                    line=dict(color='#1f77b4')
                ))

                # Predicted data
                fig_trend.add_trace(go.Scatter(
                    x=predictions['timestamp'],
                    y=predictions['predicted_sentiment'],
                    name='Predicted',
                    line=dict(color='#ff7f0e', dash='dash')
                ))

                fig_trend.update_layout(
                    title="Sentiment Trend and Prediction",
                    xaxis_title="Time",
                    yaxis_title="Sentiment Score"
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            with col2:
                st.subheader("Model Performance")
                performance = trend_results['model_performance']
                st.metric("Model Confidence", f"{performance['confidence']:.2%}")
                st.metric("Training Score", f"{performance['train_score']:.2%}")
                st.metric("Test Score", f"{performance['test_score']:.2%}")

                st.info("""
                    The prediction model uses historical sentiment patterns,
                    time-based features, and rolling statistics to forecast
                    future sentiment trends. Higher confidence scores indicate
                    more reliable predictions.
                """)
else:
    st.info("Need at least 10 data points for trend prediction. Add more data to see predictions.")


# Historical Data Table
st.header("📊 Historical Data")
st.dataframe(df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit, TextBlob, OpenAI, and spaCy")