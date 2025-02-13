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
        def get_sentiment(row):
            try:
                rating = float(row['rating'])
                text = str(row['review_content'])
                text_sentiment = TextBlob(text).sentiment.polarity
                
                # 评分分类
                # >= 4.8: 直接判定为正面
                # 3.0-4.7: 需要结合文本分析
                # < 3.0: 直接判定为负面
                if rating >= 4.8:
                    return 1.0  # 直接返回正面
                elif rating < 3.0:
                    return -1.0  # 直接返回负面
                
                # 3.0-4.7分需要结合文本分析
                if rating >= 4.0:
                    # 4.0-4.7分
                    if text_sentiment < -0.1:  # 降低负面阈值
                        return -1.0
                    elif text_sentiment > 0.4:  # 提高正面阈值
                        return 1.0
                    else:
                        return 0.0
                else:
                    # 3.0-3.9分，倾向于负面
                    if text_sentiment > 0.4:  # 需要很强的正面评价
                        return 1.0
                    elif text_sentiment < -0.1:  # 轻微负面就算负面
                        return -1.0
                    else:
                        return -1.0  # 默认为负面
                
            except:
                return 0

        df['sentiment'] = df.apply(get_sentiment, axis=1)
        return df
    
    def get_frequent_words(self, texts, n=10, sentiment_type='positive'):
        """获取高频情感词"""
        words = []
        word_sentiments = {}  # 用于缓存词的情感值
        
        for text in texts:
            if isinstance(text, str):
                # 转换为小写并分词
                text_words = re.findall(r'\w+', text.lower())
                # 过滤停用词和数字
                text_words = [word for word in text_words 
                            if word not in self.stop_words
                            and not word.isdigit()
                            and len(word) > 2]  # 过滤短词
                
                # 获取整个评论的情感
                text_sentiment = TextBlob(text).sentiment.polarity
                
                # 根据评论的整体情感来判断词的情感
                for word in text_words:
                    if word not in word_sentiments:
                        # 查找包含这个词的短语
                        word_context = re.findall(f'[a-z]* {word} [a-z]*', text.lower())
                        if word_context:
                            # 使用短语的情感
                            context_sentiment = sum(TextBlob(phrase).sentiment.polarity 
                                                 for phrase in word_context) / len(word_context)
                        else:
                            # 如果找不到短语，使用评论的整体情感
                            context_sentiment = text_sentiment
                        
                        word_sentiments[word] = context_sentiment
                    
                    # 根据情感类型筛选词
                    if (sentiment_type == 'positive' and word_sentiments[word] > 0.1) or \
                       (sentiment_type == 'negative' and word_sentiments[word] < -0.1):
                        words.append(word)
        
        # 统计词频
        word_freq = Counter(words)
        # 按情感强度排序
        if sentiment_type == 'positive':
            sorted_words = sorted(word_freq.items(), 
                                key=lambda x: (x[1], word_sentiments[x[0]]), 
                                reverse=True)
        else:
            sorted_words = sorted(word_freq.items(), 
                                key=lambda x: (x[1], -word_sentiments[x[0]]), 
                                reverse=True)
        
        return sorted_words[:n]
    
    def generate_word_cloud(self, df):
        """生成词云图"""
        positive_reviews = df[df['sentiment'] > 0]['review_content']
        negative_reviews = df[df['sentiment'] < 0]['review_content']
        
        self.positive_words = self.get_frequent_words(positive_reviews)
        self.negative_words = self.get_frequent_words(negative_reviews)