# SmcpNet
このリポジトリには、PyTorchによるSmcpNetの実装が含まれており、次のものが含まれます.

- コード進行に基づいたジャズ音楽を学習、生成を行うモデル

- 学習済みモデル

- 学習済みモデルによる音楽生成の結果

SmcpNetをルートとしてすべてのスクリプトを実行します。
即興演奏の生成のみに関心がある場合は、生成結果に進んでください。

## 動作環境
- Python 3.8.5

- CUDA 10.1

## 環境構築
```
$ pip install -r requirements.txt
```

## データの準備
xml形式の楽曲データからcsv形式のデータセットを作成する。
```
$ python Data2Dataset.py --data  xmlデータのフォルダのパス --output データセットを出力するパス
```

```
#(例)
$ python Data2Dataset.py --data data/charlie/ --output dataset/charlie/
```
- xmlデータの場所
data/データ名/に学習に使用するデータを配置。Data2Datset.pyを実行するとdataset/データセット名/にデータセットが保存される。


<pre> 
.  
└───data  
│    └データ名  
│          │   file1.xml  
│          │   file2.xml  
│          │   file3.xml  
│          │     
│
└───dataset  
│       └データセット名  
│          │   file1.csv  
│          │   file2.csv    
│          │   file3.  
│          │   
</pre>

- パラメータ
    - --data 入力するデータフォルダのパス。

    - --output 出力するデータセットフォルダへのパス。

## モデルの学習
データセットの学習を行う。
```
$ python Train.py --data データセットのフォルダのパス --LogName 学習結果を保存するフォルダ名
```

```
#(例)
$ python Train.py --data dataset/Charlie/ --LogName charlie
```
- 学習モデルは./function/train/network.pyで定義。

- 学習結果は./logに保存

- パラメータ
    - --data 入力するデータセットフォルダのパス。

    - --LogName 保存する学習結果のフォルダ名

    - --cuda cuda:GPU使用。cpu:CPU使用


## モデルによる生成
データセットの学習を行う。
```
$ python Generate.py --PathName 学習結果のフォルダ名 --ModelName 出力するフォルダ名 --DatasetPath 学習に使用したデータセットのパス --DataSelect 出力結果を選択する関数名 --RuleName 生成時のルール名
```

```
#(例)
$ python Generate.py --PathName charlie --ModelName generate --DatasetPath dataset/charlie --RuleName model_only
```
- 生成結果は./data_generateに保存

- パラメータ
    - --PathName 保存した学習結果のフォルダ名

    - --ModelName 保存する生成結果のフォルダ名

    - --DatasetPath 使用したデータセットフォルダのパス。

    - --RuleName
        - model_only
            
            深層学習のモデルのみで音楽を生成。

        - model_searchTree
        
            深層学習のモデルに木探索を組み合わせて、音楽を生成。

## 生成結果
論文の生成結果は./data_generate/sample/にxml形式で保存されています。

## Authors
Ryo Ogasawara

## References
