import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import re
import spacy

nlp = spacy.load("ja_core_news_sm")

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

# モデルとCountVectorizerの読み込み
model = joblib.load('models/trained_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')  # 事前に保存しておく必要があります

# 新しいデータの読み込み
new_data = pd.read_csv('data/new_data/new_data.csv')['text']

for te in new_data:
    print("テキスト",te)
    # 新しいテキストをトークン化し、前処理
    pro_text = spacy_tokenizer(te)    
    # トークンのリストをスペースで結合して文字列に戻す
    pro_text_joined = " ".join(pro_text)
    # 新しいデータのベクトル化
    new_X = vectorizer.transform([pro_text_joined])
    #print("ベクトルデータ:", new_X.toarray())  # スパースマトリックスを密な配列に変換して表示
    # 新しいデータのトピック分布の予測
    doc_topic_dist = model.transform(new_X)

    max_topic_index = doc_topic_dist.argmax(axis=1)[0]
    max_topic_value = doc_topic_dist[0, max_topic_index]

    # トピックインデックスとその値を出力
    print(f"Text: {te}")
    print(f"Most likely topic: {max_topic_index}, Value: {max_topic_value:.4f}\n")
    
"""
# あるいは
new_data['topic'] = doc_topic_dist.argmax(axis=1)
new_data.to_csv('data/new_data/predicted_data.csv', index=False)

"""