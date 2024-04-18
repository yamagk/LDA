import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from guidedlda import GuidedLDA
import numpy as np
import joblib
import re
import spacy

nlp = spacy.load("ja_core_news_sm")

# データの読み込み
train_data = pd.read_csv('data/training_data/train_data.csv')['text']


def spacy_tokenizer(text):
    tokens = []
    for tok in nlp(text):
        if re.match(r'^[\u3041-\u3096\u3099-\u309E]$', tok.text) or len(tok.text) == 1 or tok.is_punct:
            continue
        if re.match(r'^[\u3041-\u3096\u3099-\u309E]{2}$', tok.text):
            continue
        if tok.pos_ == "VERB":
            tokens.append(tok.lemma_)
        else:
            tokens.append(tok.text)
    return tokens

# トークン化とベクトル化
processed_texts = [spacy_tokenizer(text) for text in train_data]
processed_texts_joined = [" ".join(text) for text in processed_texts]
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(processed_texts_joined)
vocab = vectorizer.get_feature_names_out()
word2id = dict((v, idx) for idx, v in enumerate(vocab))

# モデルのトレーニング
X = vectorizer.fit_transform(processed_texts_joined)
vocab = vectorizer.get_feature_names_out()
word2id = dict((v, idx) for idx, v in enumerate(vocab))
print("Vocabulary:")

# 教師データ
seed_topic_list = {
    "寄付": 0, "良い": 0, "美しい": 0, "明るい": 0, "素晴らしい": 0, "助ける": 0, "保護": 0, 
    "粉飾": 1, "脱税": 1, "インサイダー": 1, "殺人": 1, "疑惑": 1, "横領": 1, "逮捕": 1, "検挙": 1, "捜査": 1, "判決": 1,
    "世界": 2, "休憩": 2, "写真": 2, "新作": 2, "日常": 2, "昨日": 2, "毎日": 2, "生活": 2, "音楽": 2, "カフェ": 2, "プロジェクト": 2, "コーヒー" : 2,
    "プライバシー": 3, "ポリシー": 3,"セキュリティ-": 3,"方針": 3, "保護": 3, "コンプライアンス": 3,
    "テレビ": 4,"ドラマ": 4,"創作": 4,"映画" :4,"劇場": 4,"ストーリー": 4,"文庫": 4,"本": 4
    }
seed_topics = {}
for word, topic_num in seed_topic_list.items():
    if word in word2id:
        seed_topics[word2id[word]] = topic_num
    else:
        print(f"Warning: '{word}' not found in vocabulary.")


model = GuidedLDA(n_topics=5, n_iter=100, random_state=10, refresh=20)
model.fit(X, seed_topics=seed_topics, seed_confidence=0.85)

# モデルの保存
joblib.dump(model, 'models/trained_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

# 各トピックの単語分布を表示
n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
