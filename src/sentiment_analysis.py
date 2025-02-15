import pandas as pd
from transformers import pipeline
import torch

def analyze_reviews():
    """分析评论情感"""
    try:
        # 加载数据
        print("=== Loading Data ===")
        df = pd.read_csv('data/processed_amazon.csv')
        
        # 初始化BERT模型
        print("Initializing BERT model...")
        classifier = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 分析评论
        print("\nAnalyzing reviews...")
        results = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(df)} reviews...")
                
            if pd.isna(row['cleaned_review']) or len(str(row['cleaned_review']).strip()) == 0:
                result = {'label': 'NEUTRAL', 'score': 0.5}
            else:
                result = classifier(str(row['cleaned_review'])[:512])[0]
            
            results.append(result)
        
        # 添加结果到数据框
        df['sentiment'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        
        # 计算情感分布
        total = len(df)
        positive = sum(df['sentiment'] == 'POSITIVE')
        negative = sum(df['sentiment'] == 'NEGATIVE')
        
        print("\n=== Sentiment Analysis Results ===")
        print(f"Total reviews: {total}")
        print(f"Positive: {positive} ({positive/total*100:.1f}%)")
        print(f"Negative: {negative} ({negative/total*100:.1f}%)")
        print(f"Ratio (Positive:Negative) = {positive}:{negative} ({positive/negative:.2f}:1)")
        print(f"Average sentiment score: {df['sentiment_score'].mean():.2f}")
        
        # 保存结果
        df.to_csv('data/processed_amazon.csv', index=False)
        print("\nResults saved to processed_amazon.csv")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    analyze_reviews()