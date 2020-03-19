# 再現手順
## ライブラリをインストール

numpy

	$ pip install numpy

pandas

	$ pip install pandas

matplotlib

	$ pip install matplotlib

seaborn

	$ pip install seaborn

sklearn

	$ pip install scikit-learn

mecab 

      $ brew install mecab mecab-ipadic git curl xz

      $ git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git

      $ cd /mecab-ipadic-neologd

      $ sudo ./bin/install-mecab-ipadic-neologd -n -a

      $ echo `mecab-config --dicdir`"/mecab-ipadic-neologd" （階層確認コマンド）

fasttext 

	 $ git clone https://github.com/facebookresearch/fastText.git

	 $ cd fastText/

	 $ make

	 $ pip install fasttext

## 学習済みモデルをダウンロード

https://fasttext.cc/docs/en/crawl-vectors.html

-> 下にスクロールして，Japaneseの部分のbinをクリックしダウンロード

## 実行

	$ python3 fthanler.py 