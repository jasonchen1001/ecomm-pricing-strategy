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
        
        # 添加产品相关的中性词（这些词不表达情感）
        self.neutral_words = {
            # 产品特征词
            'cable', 'charger', 'charging', 'wire', 'cord', 'adapter', 'port',
            'usb', 'type', 'power', 'device', 'phone', 'data', 'length', 'meter',
            
            # 品牌词
            'amazon', 'brand', 'company', 'seller',
            
            # 时间和数量词
            'time', 'day', 'month', 'year', 'piece', 'pack', 'size',
            
            # 常见动词
            'use', 'using', 'used', 'buy', 'bought', 'purchase', 'ordered',
            
            # 其他中性词
            'product', 'price', 'cost', 'review', 'rating'
        }
        
        # 积极情感词典
        self.positive_words = {
            # 质量相关
            'excellent', 'perfect', 'best', 'premium', 'superior',
            'quality', 'durable', 'sturdy', 'reliable', 'solid',
            
            # 性能相关
            'fast', 'quick', 'rapid', 'efficient', 'effective',
            'powerful', 'strong', 'stable', 'smooth', 'seamless',
            
            # 评价相关
            'amazing', 'awesome', 'fantastic', 'great', 'wonderful',
            'satisfied', 'happy', 'impressed', 'recommended', 'worth',
            'good', 'nice', 'love', 'perfect', 'excellent'
        }
        
        # 消极情感词典
        self.negative_words = {
            # 质量相关
            'poor', 'bad', 'terrible', 'horrible', 'cheap',
            'defective', 'faulty', 'broken', 'damaged', 'fragile',
            'low', 'poor', 'inferior', 'flimsy', 'loose',
            'break', 'breaks', 'breaking', 'broke',
            
            # 性能相关
            'slow', 'weak', 'unstable', 'inconsistent', 'unreliable',
            'fail', 'failed', 'failing', 'fails', 'failure',
            'stop', 'stops', 'stopped', 'stopping',
            'disconnect', 'disconnects', 'disconnected',
            'error', 'errors', 'problem', 'problems',
            
            # 评价相关
            'disappointed', 'disappointing', 'disappointment',
            'waste', 'wasted', 'wasting',
            'avoid', 'avoided', 'avoiding',
            'regret', 'regrets', 'regretted',
            'return', 'returned', 'returning',
            'refund', 'refunded', 'refunding',
            'complaint', 'complaints', 'complaining',
            'issue', 'issues', 'problem', 'problems',
            'worse', 'worst', 'bad', 'badly',
            'expensive', 'overpriced', 'costly',
            'not worth', 'worthless', 'useless',
            
            # 其他负面词
            'difficult', 'hard', 'trouble', 'troublesome',
            'poor', 'cheap', 'cheaply', 'inferior',
            'hate', 'hated', 'hating', 'dislike',
            'angry', 'anger', 'frustrated', 'frustrating',
            'annoying', 'annoyed', 'irritating', 'irritated',
            'disappointed', 'disappointing', 'disappointment',
            'wrong', 'incorrect', 'improper', 'inappropriate',
            'damage', 'damaged', 'damaging', 'damages'
        }
        
        # 更新停用词，加入中性词
        self.stop_words.update(self.neutral_words)
    
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
        """获取高频情感词
        
        Args:
            texts: 评论文本列表
            n: 返回的高频词数量
            sentiment_type: 'positive' 或 'negative'，指定要统计的情感类型
        """
        words = []
        # 选择目标情感词典
        target_words = self.positive_words if sentiment_type == 'positive' else self.negative_words
        
        for text in texts:
            if isinstance(text, str):
                # 转换为小写并分词
                text_words = re.findall(r'\w+', text.lower())
                # 只保留目标情感词典中的词
                text_words = [word for word in text_words 
                             if word in target_words
                             and not word.isdigit()
                             and len(word) > 2]  # 过滤短词
                words.extend(text_words)
        
        # 统计词频
        word_freq = Counter(words)
        return word_freq.most_common(n)
    
    def generate_word_cloud(self, df):
        """生成词云图"""
        positive_reviews = df[df['sentiment'] > 0]['review_content']
        negative_reviews = df[df['sentiment'] < 0]['review_content']
        
        self.positive_words = self.get_frequent_words(positive_reviews)
        self.negative_words = self.get_frequent_words(negative_reviews) 