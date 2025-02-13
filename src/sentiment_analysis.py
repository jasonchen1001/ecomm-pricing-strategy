from textblob import TextBlob
from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

class SentimentAnalyzer:
    def __init__(self):
        # 下载必要的NLTK数据
        try:
            nltk.data.find('corpora/stopwords')
            nltk.download('punkt')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
        
        # 获取英文停用词
        self.stop_words = set(stopwords.words('english'))
        # 添加产品相关的中性词
        self.stop_words.update({
            'cable', 'charger', 'wire', 'cord', 'adapter', 'device',
            'amazon', 'product', 'price', 'review', 'star', 'rating',
            'buy', 'bought', 'purchase', 'ordered', 'received',
            'use', 'using', 'used', 'time', 'month', 'day', 'year'
        })
    
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
    
    def get_frequent_words(self, texts, n=10, sentiment_type='positive'):
        """获取高频情感词"""
        words = []
        
        for text in texts:
            if isinstance(text, str):
                # 转换为小写并分词
                text_words = re.findall(r'\w+', text.lower())
                # 过滤停用词和数字
                text_words = [word for word in text_words 
                            if word not in self.stop_words
                            and not word.isdigit()
                            and len(word) > 2]  # 过滤短词
                
                # 使用 TextBlob 判断每个词的情感
                for word in text_words:
                    sentiment = TextBlob(word).sentiment.polarity
                    # 根据情感类型筛选词
                    if (sentiment_type == 'positive' and sentiment > 0) or \
                       (sentiment_type == 'negative' and sentiment < 0):
                        words.append(word)
        
        # 统计词频
        word_freq = Counter(words)
        return word_freq.most_common(n)
    
    def generate_word_cloud(self, df):
        """生成词云图"""
        positive_reviews = df[df['sentiment'] > 0]['review_content']
        negative_reviews = df[df['sentiment'] < 0]['review_content']
        
        self.positive_words = self.get_frequent_words(positive_reviews)
        self.negative_words = self.get_frequent_words(negative_reviews)