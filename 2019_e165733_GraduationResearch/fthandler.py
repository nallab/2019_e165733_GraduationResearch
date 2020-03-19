# coding: utf-8
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MeCab
import fasttext as ft
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def get_tokens(content):
    # 辞書を適用
    tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    # ストップワードを定義
    stop_words = set(
        '——!? —？ \ ＝ ＜ ＞ < > 〈 〉 ― ー ─ ！ ! ？ - & ―― % \ ( \ ) \ . \ , / = \ - ， ． 。 、 \  ￥ ’ “ ” " → ↓ ← ↑ … 「 」  （ ） ー \  ～ 『  \ ・ ～ 『 』 ・ ； ： ※ ％ )」 ) )〉 )。 …。 【  】'.split())

    # 文を単語に分割し、ストップワードを除去
    tokens = [word for word in tagger.parse(content).strip().split() if word not in stop_words]

    return tokens

def set_df_text(dataframe, model):
    # テキストのデータフレームを整形してベクトル追加
    if 'body' not in dataframe.columns:
        print(u'文書がありません。body の名前でカラムを作成してください。')
        return

    else:
        df_text = dataframe.copy()
        if 'vector' not in df_text.columns:
            df_text['vector'] = None

        df_text['body'] = df_text['body'].str.replace(r'<[^>]*>|\r|\n', ' ')
        df_text['body'] = df_text['body'].str.replace(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', ' ')
        df_text['vector'] = [
            text2vec(body, model, consider_length=False) for body
            in df_text['body'].astype('str').values]

        return df_text


def text2vec(text, model, consider_length):
    return list(combine_word_vectors(get_tokens(text), model, is_mean=consider_length))


def combine_word_vectors(words, model, is_mean):
    # modelをloadしていない場合は警告
    if model is None:
        print('Model is not set.')
        print('Try `cls.load` or `cls.generate_model`.')
        return

    # 指定された関数を適用してベクトルを生成
    apply_func = np.mean if is_mean else np.sum

    return apply_func(list(model.get_word_vector(x) for x in words), axis=0)


def ft_vector(input_file, model, dim):
    plots = []
    mecab = []
    # csvファイルをを1行ずつ読み込み配列に格納
    with open(input_file, 'r') as f:
        data = csv.reader(f)
        for row in data:
            plots.append(row)
    dataframe = pd.DataFrame(plots, columns=['body', 'class_id'])

    # 配列のテキスト部分を形態素解析
    for data in dataframe['body']:
        mecab.append(get_tokens(data))

    # テキストのベクトル化
    df_tmp = set_df_text(dataframe, model)

    # ベクトルの次元を圧縮
    df_tmp = dimension_reduction(df_tmp, dim)

    # 不要なカラムを削除
    del df_tmp['body']
    del df_tmp['class_id']

    # 次元削除後のベクトルを格納したカラムを追加
    df_vecadd = pd.merge(dataframe, df_tmp, how='left',
                         left_index=True, right_index=True)

    df_vecadd['body'] = mecab

    return df_vecadd


def dimension_reduction(data, pca_dimension):
    # 文章ベクトルの次元圧縮
    pca_data = data.copy()
    pca = PCA(n_components=pca_dimension)
    vector = np.array([np.array(v) for v in pca_data['vector']])
    pca_vector = pca.fit_transform(vector)
    pca_data['pca_vector'] = [v for v in pca_vector]
    del pca_data['vector']
    pca_data.rename(columns={'pca_vector': 'vector'}, inplace=True)

    return pca_data


def calassifier(train_data):
    tokens = train_data['body']
    result_list = []
    result_list_int = []
    y_int = []
    correct_y0 = []
    correct_y1 = []
    miss_y0 = []
    miss_y1 = []

    # ベクトルカラムとラベルカラムをx,yにnumpyで格納
    x = np.array([np.array(v) for v in train_data['vector']])
    y = np.array([i for i in train_data['class_id']])

    # ガウシアンナイーブベイズ分類器の呼び出し
    clf = GaussianNB()

    # Leave-one-outのインスタンス生成
    loo = LeaveOneOut()

    # Leave-one-outの実装
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(x_train, y_train) # 学習データで学習する
        result = clf.predict(x_test)  # テストデータからラベルを予測する
        result_list.append(result)  # 予測したラベルを格納
        result_list_int.append(result.astype(np.int32)) # 予測したラベルをint型で格納
        y_int.append(y_test.astype(np.int32)) # 正解ラベルをint型で格納
        #　2値分類(0,1)の正予測，誤予測の4パターンをそれぞれ違うラベルに格納
        if result == y_test:
            if y_test == '0':
                correct_y0.append(tokens[int(test_index)])
            else:
                correct_y1.append(tokens[int(test_index)])
        else:
            if y_test == '0':
                miss_y0.append(tokens[int(test_index)])
            else:
                miss_y1.append(tokens[int(test_index)])

    rate = accuracy_score(y, result_list)
    cm = confusion_matrix(y, result_list)

    sns.heatmap(confusion_matrix(y, result_list), annot=True)
    plt.savefig('ft_naive_bayes.png')

    print("正解したラベル0")
    for cy0 in correct_y0:
        print(cy0)
    print('\n')

    print("間違ったラベル0")
    for my0 in miss_y0:
        print(my0)
    print('\n')

    print("正解したラベル1")
    for cy1 in correct_y1:
        print(cy1)
    print('\n')

    print("間違ったラベル1")
    for my1 in miss_y1:
        print(my1)
    print('\n')

    return rate, cm

def main():
    # 学習済みモデルを読み込み
    ft_model = ft.load_model('cc.ja.300.bin')
    # 次元削除後の次元数を設定
    vector_dim = 8
    # テキストデータを読み込み前処理
    train_data = ft_vector('data.csv', ft_model, dim=vector_dim)
    # 分後し，正答率と混合行列を格納
    rate, cm = calassifier(train_data)
    print({'次元数': vector_dim, '正答率': rate})
    print(cm)

if __name__ == '__main__':
    main()
