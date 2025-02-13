from textblob import TextBlob
from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

class SentimentAnalyzer:
    def __init__(self):
        # 下载必要的NLTK数据
        try:
            nltk.data.find('corpora/stopwords')
            nltk.download('punkt')
            nltk.download('vader_lexicon')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('vader_lexicon')
        
        # 初始化VADER分析器
        self.sia = SentimentIntensityAnalyzer()
        
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
        """使用VADER分析文本情感"""
        try:
            scores = self.sia.polarity_scores(str(text))
            return scores['compound']  # 返回复合得分 [-1, 1]
        except:
            return 0
    
    def analyze_reviews(self, df):
        """分析所有评论"""
        def get_sentiment(row):
            try:
                rating = float(row['rating'])
                text = str(row['review_content'])
                text_sentiment = self.analyze_text(text)
                
                # 1. 评分判定
                if rating <= 3.0:  # 3星及以下
                    return -1.0  # 直接判定为负面
                
                # 2. 文本判定（对于3星以上的评论）
                # 如果文本明显表达不满，即使评分高也判定为负面
                if len(text) > 10:  # 确保评论有足够长度
                    # 检查负面关键词
                    negative_keywords = ['bad', 'poor', 'worst', 'terrible', 'waste', 
                                      'not good', 'not worth', 'disappointed', 'broke',
                                      'stopped working', 'cheap quality', 'don\'t buy']
                    text_lower = text.lower()
                    
                    # 如果包含强烈的负面词，判定为负面
                    if any(keyword in text_lower for keyword in negative_keywords):
                        return -1.0
                    
                    # 使用VADER的详细得分
                    scores = self.sia.polarity_scores(text)
                    
                    # 如果负面得分显著，判定为负面
                    if scores['neg'] > 0.2:  # 负面成分超过20%
                        return -1.0
                    
                    # 中性判定：如果正面和负面成分都不明显
                    if scores['pos'] < 0.2 and scores['neg'] < 0.1:
                        return 0.0
                
                # 3. 评分和文本综合判定
                if rating >= 4.5:  # 4.5星以上
                    if text_sentiment > 0:  # 文本情感为正
                        return 1.0
                    else:
                        return 0.0  # 文本不够正面，判为中性
                elif rating >= 4.0:  # 4-4.5星
                    if text_sentiment > 0.2:  # 需要明显的正面文本
                        return 1.0
                    else:
                        return 0.0
                else:  # 3-4星
                    if text_sentiment > 0.4:  # 需要很强的正面文本
                        return 1.0
                    elif text_sentiment < -0.2:  # 稍微负面就算负面
                        return -1.0
                    else:
                        return 0.0
                
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                return 0

        df['sentiment'] = df.apply(get_sentiment, axis=1)
        
        # 打印分布情况用于调试
        sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
        print("\nSentiment Distribution:")
        print(f"Positive: {sentiment_dist.get(1.0, 0):.1f}%")
        print(f"Neutral: {sentiment_dist.get(0.0, 0):.1f}%")
        print(f"Negative: {sentiment_dist.get(-1.0, 0):.1f}%")
        
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