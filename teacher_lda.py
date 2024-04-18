import guidedlda
import spacy
import numpy as np
import re

# spacyの日本語モデルをロード
nlp = spacy.load("ja_core_news_sm")

# 文書の頻度をベクトル化する必要があるため。一連の処理    
from sklearn.feature_extraction.text import CountVectorizer

def spacy_tokenizer(text):
    tokens = []
    for tok in nlp(text):
        # 1文字のひらがな、濁点付きも含む、および記号を除去
        if re.match(r'^[\u3041-\u3096\u3099-\u309E]$', tok.text) or len(tok.text) == 1 or tok.is_punct:
            continue
        # 2文字のひらがなを除去
        if re.match(r'^[\u3041-\u3096\u3099-\u309E]{2}$', tok.text):
            continue
        # 動詞を基本形に変換、それ以外はそのまま使用
        if tok.pos_ == "VERB":
            tokens.append(tok.lemma_)
        else:
            tokens.append(tok.text)
    return tokens


text_all_kanma = [
    #ポジティブ
    "彼の寄付は多くの人々を助けました。",
    "新しい公園は地域社会にとって素晴らしい資源です",
    "ボランティア団体は被災者支援で大きな違いをもたらしています。",
    "清掃活動はこのビーチを再び美しい場所にしました。",
    "彼女の研究は環境保護に大きく貢献しています。",
    "地元の農家から食材を仕入れることで、サステナブルな消費を推進しています。",
    "エコフレンドリーな製品は、環境への影響を減らすのに役立ちます。",
    "この慈善団体は世界中の貧困を減らすために尽力しています。",
    "社員の健康増進プログラムは大きな成功を収めています。",
    "彼は地域社会のために無料の教育プログラムを提供しています。",
    "新しいカフェのコーヒーが美味しいです。",
    "友人との再会がとても楽しかった。",
    "最近読んだ本が非常に感動的だった。",
    "今日の天気はとても良い。",
    "昨日のプレゼンテーションは成功だった。",
    "彼の笑顔が明るい一日の始まりを告げる。",
    "このアプリは日常生活に便利で使いやすい。",
    "散歩するのに最適な公園を見つけた。",
    "映画館で見た新作映画が素晴らしかった。",
    "家族との夕食が毎日の楽しみです。",
    "休暇で訪れたビーチが美しかった。",
    "新しいレシピの試作が成功し、大満足です。",
    "古い友人からの手紙を受け取り、嬉しかった。",
    "新しい趣味の写真撮影が楽しい。",
    "朝の散歩で心が落ち着く。",
    "子供たちの成長が目を見張るものがある。",
    "地元のイベントで素敵な出会いがあった。",
    "この冬は暖かく過ごせそうです。",
    "音楽を聴きながらリラックスする時間が好き。",
    "ガーデニングが私のストレス解消法です。",
#ネガティブ
    "その企業は重大な環境汚染事件を引き起こしました。",
    "彼は詐欺で多くの人からお金を騙し取りました。",
    "不正アクセスで顧客のデータが流出した。",
    "その政治家は脱税の疑いで捜査されています。",
    "殺人事件の犯人はまだ逮捕されていません。",
    "インサイダー取引によって市場が不正に操作された。",
    "反社会的勢力との関連が疑われている。",
    "不正な手段で選挙に介入しようとした疑いがある。",
    "会社の経理から大金が横領された。",
    "偽造された商品が市場に出回っている。",
     "会社の粉飾決算が発覚し、業界全体に衝撃を与えた。",
    "役員が横領疑惑で調査されている。",
    "着服された資金の額が膨大であることが判明した。",
    "背任行為により、重要なプロジェクトが遅延している。",
    "脱税疑惑で有名企業がニュースになった。",
    "申告漏れが指摘され、税務調査が入った。",
    "違反行為により、彼は職を失った。",
    "逮捕された後、そのスキャンダルはさらに大きくなった。",
    "送検された事件が社会問題となっている。",
    "検挙されたグループは長年にわたる犯罪歴がある。",
    "捜査が進む中で、新たな証拠が見つかった。",
    "捜索中に隠された資金が見つかった。",
    "指名手配された犯人はまだ逃亡中だ。",
    "判決が下され、厳しい罰が言い渡された。",
    "提訴された会社は、法廷で争うことになる。",
    "告訴により、事件の調査が再開された。",
    "彼はその罪を否定している。",
    "詐欺事件に巻き込まれ、多くの人が被害を受けた。",
    "不正な手段による利益は、最終的には失われる。",
    "偽装された商品が市場に出回っているとの報告がある。"
]

processed_texts = [spacy_tokenizer(text) for text in text_all_kanma]
processed_texts_joined = [" ".join(text) for text in processed_texts]

# CountVectorizerをカスタムトークン化関数で初期化
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(processed_texts_joined)
vocab = vectorizer.get_feature_names_out()
word2id = dict((v, idx) for idx, v in enumerate(vocab))
print("Vocabulary:", vocab)

# シードトピックのリストを作成
seed_topic_list = {
    "寄付": 0, "良い": 0, "美しい": 0, "明るい": 0, "素晴らしい": 0, "助ける": 0, "保護": 0, 
    "粉飾": 1,  "脱税": 1, "インサイダー": 1, "殺人": 1, "疑惑": 1, "横領": 1, "逮捕": 1, "検挙": 1, "捜査": 1, "判決": 1
}

# 数値IDの辞書を作成
seed_topics = {}
for word, topic_num in seed_topic_list.items():
    seed_topics[word2id[word]] = topic_num

# GuidedLDAモデルのインスタンスを作成
model = guidedlda.GuidedLDA(n_topics=2, n_iter=100, random_state=7, refresh=20)

# モデルにデータをフィットさせる
model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)

# 各トピックの単語分布を表示
n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


new_documents = ["殺人事件で逮捕された犯罪者"]
new_doc_term_matrix = vectorizer.transform(new_documents)

# LDAモデルを使ってトピック分布を推定する
new_doc_topic_dist = model.transform(new_doc_term_matrix)

# 新しい文書のトピック分布を出力する
print(new_doc_topic_dist)