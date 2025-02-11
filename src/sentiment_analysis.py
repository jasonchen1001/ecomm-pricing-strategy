from textblob import TextBlob
from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt

class SentimentAnalyzer:
    def __init__(self):
        self.positive_words = []
        self.negative_words = []
    
    def analyze_text(self, text):
        """分析单条评论的情感"""
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0
    
    def analyze_reviews(self, df):
        """分析所有评论"""
        df['sentiment'] = df['review_content'].apply(self.analyze_text)
        return df
    
    def get_frequent_words(self, texts, n=10):
        """获取高频词"""
        words = []
        for text in texts:
            if isinstance(text, str):
                words.extend(re.findall(r'\w+', text.lower()))
        return Counter(words).most_common(n)
    
    def generate_word_cloud(self, df):
        """生成词云图"""
        positive_reviews = df[df['sentiment'] > 0]['review_content']
        negative_reviews = df[df['sentiment'] < 0]['review_content']
        
        self.positive_words = self.get_frequent_words(positive_reviews)
        self.negative_words = self.get_frequent_words(negative_reviews) 