import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def prepare_time_features(df):
    """
    Prepare time-based features for the model.
    """
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    # Calculate rolling statistics
    df['rolling_avg_sentiment'] = df['polarity'].rolling(window=5, min_periods=1).mean()
    df['sentiment_std'] = df['polarity'].rolling(window=5, min_periods=1).std()
    
    return df

def train_prediction_model(df):
    """
    Train a model to predict sentiment trends.
    """
    # Prepare features
    df = prepare_time_features(df)
    
    # Prepare feature matrix
    features = ['hour', 'day_of_week', 'day_of_month', 'month', 
                'rolling_avg_sentiment', 'sentiment_std', 'polarity']
    
    # Create target variable (next period's sentiment)
    df['next_sentiment'] = df['polarity'].shift(-1)
    
    # Remove last row which will have NaN for next_sentiment
    df = df.dropna(subset=['next_sentiment'])
    
    # Split features and target
    X = df[features]
    y = df['next_sentiment']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate model performance
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    return model, scaler, train_score, test_score

def predict_future_sentiment(df, model, scaler, num_periods=7):
    """
    Predict future sentiment trends.
    """
    # Prepare the latest data
    df_copy = df.copy()
    df_copy = prepare_time_features(df_copy)
    
    # Features used for prediction
    features = ['hour', 'day_of_week', 'day_of_month', 'month', 
                'rolling_avg_sentiment', 'sentiment_std', 'polarity']
    
    predictions = []
    timestamps = []
    last_timestamp = df_copy['timestamp'].max()
    
    # Make predictions for future periods
    for i in range(num_periods):
        # Get the last row of data
        last_data = df_copy.iloc[-1:]
        
        # Create next timestamp
        next_timestamp = last_timestamp + timedelta(days=i+1)
        timestamps.append(next_timestamp)
        
        # Update time-based features for the prediction
        last_data['timestamp'] = next_timestamp
        last_data = prepare_time_features(last_data)
        
        # Prepare features for prediction
        X_pred = last_data[features]
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        prediction = model.predict(X_pred_scaled)[0]
        predictions.append(prediction)
        
        # Add prediction to dataframe for next iteration
        new_row = last_data.copy()
        new_row['polarity'] = prediction
        df_copy = pd.concat([df_copy, new_row])
    
    # Create prediction results
    prediction_df = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_sentiment': predictions
    })
    
    return prediction_df

def get_trend_analysis(df):
    """
    Get trend analysis including predictions and model performance metrics.
    """
    if len(df) < 10:  # Need minimum data points for meaningful prediction
        return {
            'error': 'Insufficient data for trend prediction. Need at least 10 data points.',
            'predictions': None,
            'model_performance': None
        }
    
    try:
        # Train model
        model, scaler, train_score, test_score = train_prediction_model(df)
        
        # Make predictions
        predictions = predict_future_sentiment(df, model, scaler)
        
        # Calculate confidence based on model performance
        confidence = (train_score + test_score) / 2
        
        return {
            'error': None,
            'predictions': predictions,
            'model_performance': {
                'train_score': train_score,
                'test_score': test_score,
                'confidence': confidence
            }
        }
    except Exception as e:
        return {
            'error': f'Error in trend prediction: {str(e)}',
            'predictions': None,
            'model_performance': None
        }
