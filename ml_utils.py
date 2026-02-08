import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (only needs to happen once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def predict_sentiment(text):
    """
    Analyzes sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Returns: 'positive', 'negative', or 'neutral'
    """
    analyzer = SentimentIntensityAnalyzer()
    
    # here Compound score ranges from -1 (most extreme negative) to +1 (most extreme positive) and is pretty good at capturing overall sentiment.
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    # Thresholds can be tuned. Standard VADER thresholds: (these are commonly used but can be adjusted based on your specific use case)
    # positive: compound >= 0.05 and  neutral:  (compound > -0.05) and (compound < 0.05)
    
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"