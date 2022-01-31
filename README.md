# CPBB-Net (English)
This repository contains PyTorch's implementation of CPBB-Net. It also contains the following

- A model for learning and generating jazz music based on chord progressions.

- Trained model

- Results of music generation using the trained model

Run all scripts with CPBB-Net as root.
If you are only interested in generating improvisations, go to Generated Results.

## Operating Environment
- Python 3.8.5

- CUDA 10.1

## Environment Creation
```
$ pip install -r requirements.txt
```

## Prepare the data
Create a csv-format data set from xml-format music data.
However, the melody and chord information is required for the music data in xml format
In this study, we use the xml data from <https://homepages.loria.fr/evincent/omnibook/>.
```
$ python Data2Dataset.py --data  [Path of the xml data folder] --output [Path to output the dataset]
```

```
# Example
$ python Data2Dataset.py --data data/test/ --output dataset/test/
```
- Location of the xml data
Place the data to be used for training in /data/dataset_name/. Run Data2Datset.py to save the dataset in dataset/dataset_name/.


<pre> 
.  
└───data  
│    └data-name   
│          │   file1.xml  
│          │   file2.xml  
│          │   file3.xml  
│          │     
│
└───dataset  
│       └Dataset-name   
│          │   file1.csv  
│          │   file2.csv    
│          │   file3.  
│          │   
</pre>

- Parameters
    - --data    Path of the data folder to enter.

    - --output  Path of the data folder to enter.

## Train the model.
Train the dataset.
```
$ python Train.py --data [Path of the dataset folder] --LogName [Name of the folder where the learning results will be saved]
```

```
# Example
$ python Train.py --data dataset/test/ --LogName test
```
- The learning model is defined in . /function/train/network.py.

- Learning results are saved in . /log

- Parameters
    - --data Path of the dataset folder to enter.

    - --LogName Folder name of the training results to be saved

    - --cuda cuda: Use GPU.。cpu: Use CPU.


## Generate by model.
Generate music by models.
```
$ python Generate.py --PathName [Folder name for learning results] --ModelName [Folder name for output] --DatasetPath [Path of the dataset used for training.] --RuleName [Rule name at generation]
```

```
# Example1
$ python Generate.py --PathName sample --ModelName sample --RuleName model_searchTree

# Example2(When creating rules with your own data set.)
$ python Generate.py --PathName sample --ModelName sample --Dataset2Rule_Flag --DatasetPath dataset/test --RuleName model_searchTree
```
- The generated results are saved in . /data_generate

- Parameters
    - --PathName Folder name for the saved learning results

    - --ModelName Folder name of the generated results to be saved

    - --Dataset2Rule_Flag　Used to create a rule.

    - --DatasetPath   Path of the dataset folder used (used to create the rule).

    - --RuleName
        - model_only
            
            Generate music using only deep learning models.

        - model_searchTree
        
            Music is generated by combining deep learning models with tree search.

    - --OutputExtension Extension of the generated music.
        - xml (default)

            Output in XML file format.

        - mid
        
            Output in MIDI file format.

        - both
        
            Outputs both XML and MIDI files.

    - --Tempo BPM of the generated music.(default:160)

        例 --Tempo 120
    
    - --StartUseMidiNum MIDI number of the lowest note used in the generated music.(default:0)

        例 --StartUseMidiNum 40

    - --EndUseMidiNum MIDI number of the highest note used in the generated music.(default:127)

        例 --EndUseMidiNum 110

    - --DataSelect How to select notes.
        - DataSelect (default)
        
            Select notes in the order of their model output values.

        - RandomDataSelect
        
            Notes are randomly selected based on the output values of the model.


## Generated Results

The results of the paper generation are saved in . /data_generate/SampleGenerateData/ in xml format.

## Comparison of generated results

|  Model |  Chords 1 |  Chords 2  | 
| :---: | :---: | :---: |
|  CPPB-Net+SearchTree |  <audio src="https://github.com/ryobassk/CPPB-Net/blob/master/data_generate/SampleGenerateData/mp3/Bebop_Au1.mp3" controls></audio>  |  <audio src="data_generate/SampleGenerateData/mp3/TreeCPPB_Au1.mp3" controls></audio>   | 
|  CPPB-Net  |  <audio src="data_generate/SampleGenerateData/mp3/CPPB_Au1.mp3" controls></audio> |  <audio src="data_generate/SampleGenerateData/mp3/CPPB_Ju1.mp3" controls></audio> | 
|  BeBopNet |  <audio src="data_generate/SampleGenerateData/mp3/Bebop_Au1.mp3" controls></audio>  | <audio src="data_generate/SampleGenerateData/mp3/Bebop_Ju1.mp3" controls></audio>| 

<audio controls>
    <source src="https://github.com/ryobassk/CPPB-Net/blob/master/data_generate/SampleGenerateData/mp3/Bebop_Au1.mp3">
    <source src="https://raw.githubusercontent.com/ytyaru/Audio.Sample.201708031714/master/20170803/wav/CMajor.wav">
</audio>
<audio src="https://github.com/ryobassk/CPPB-Net/blob/master/data_generate/SampleGenerateData/mp3/Bebop_Au1.mp3" controls></audio>
<figure>
    <figcaption>Listen to the T-Rex:</figcaption>
    <audio
        controls
        src="https://github.com/ryobassk/CPPB-Net/blob/master/data_generate/SampleGenerateData/mp3/Bebop_Au1.mp3">
            Your browser does not support the
            <code>audio</code> element.
    </audio>
</figure>


## Authors
Ryo Ogasawara

## References



# CPBB-Net(日本語版)
このリポジトリには、PyTorchによるCPBB-Netの実装が含まれており、次のものが含まれます.

- コード進行に基づいたジャズ音楽を学習、生成を行うモデル

- 学習済みモデル

- 学習済みモデルによる音楽生成の結果

CPBB-Netをルートとしてすべてのスクリプトを実行します。
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
但し、xml形式の楽曲データにはメロディとコードの情報が必要である
本研究では、<https://homepages.loria.fr/evincent/omnibook/>のXMLデータを使用しています。
```
$ python Data2Dataset.py --data  xmlデータのフォルダのパス --output データセットを出力するパス
```

```
#(例)
$ python Data2Dataset.py --data data/test/ --output dataset/test/
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
$ python Train.py --data dataset/test/ --LogName test
```
- 学習モデルは./function/train/network.pyで定義。

- 学習結果は./logに保存

- パラメータ
    - --data 入力するデータセットフォルダのパス。

    - --LogName 保存する学習結果のフォルダ名

    - --cuda cuda:GPU使用。cpu:CPU使用


## モデルによる生成
モデルによって音楽を生成する。
```
$ python Generate.py --PathName 学習結果のフォルダ名 --ModelName 出力するフォルダ名 --DatasetPath 学習に使用したデータセットのパス --RuleName 生成時のルール名
```

```
#(例1)
$ python Generate.py --PathName sample --ModelName sample --RuleName model_searchTree

#(例2:独自のデータセットでルールを作成するとき)
$ python Generate.py --PathName sample --ModelName sample --Dataset2Rule_Flag --DatasetPath dataset/test --RuleName model_searchTree

```
- 生成結果は./data_generateに保存

- パラメータ
    - --PathName 保存した学習結果のフォルダ名

    - --ModelName 保存する生成結果のフォルダ名

    - --Dataset2Rule_Flag　ルールを作成する時に使用

    - --DatasetPath 使用したデータセットフォルダのパス(ルール作成に使用)。

    - --RuleName
        - model_only
            
            深層学習のモデルのみで音楽を生成。

        - model_searchTree
        
            深層学習のモデルに木探索を組み合わせて、音楽を生成。

    - --OutputExtension 生成楽曲の拡張子
        - xml （デフォルト）

            XMLファイル形式で出力。

        - mid
        
            MIDIファイル形式で出力。

        - both
        
            XMLファイルとMIDIファイルのどちらも出力。

    - --Tempo 生成楽曲のBPM。(デフォルト:160)

        例 --Tempo 120
    
    - --StartUseMidiNum 生成楽曲に使用する最低音のMIDI番号。(デフォルト:0)

        例 --StartUseMidiNum 40

    - --EndUseMidiNum 生成楽曲に使用する最高音のMIDI番号。(デフォルト:127)

        例 --EndUseMidiNum 110

    - --DataSelect 音符の選択方法。
        - DataSelect （デフォルト）
        
            音符をモデルの出力値順に選択。

        - RandomDataSelect
        
            音符をモデルの出力値を元にランダムで選択。

         


## 生成結果
論文の生成結果は./data_generate/SampleGenerateData/にxml形式で保存されています。

## 生成結果の比較

|  モデル名 |  コード進行１ |  コード進行２  | 
| :---: | :---: | :---: |
|  CPPB-Net+木探索 |  <audio src="data_generate/SampleGenerateData/mp3/TreeCPPB_Au1.mp3" controls></audio>  |  <audio src="data_generate/SampleGenerateData/mp3/TreeCPPB_Au1.mp3" controls></audio>   | 
|  CPPB-Net  |  <audio src="data_generate/SampleGenerateData/mp3/CPPB_Au1.mp3" controls></audio> |  <audio src="data_generate/SampleGenerateData/mp3/CPPB_Ju1.mp3" controls></audio> | 
|  BeBopNet |  <audio src="data_generate/SampleGenerateData/mp3/Bebop_Au1.mp3" controls></audio>  | <audio src="data_generate/SampleGenerateData/mp3/Bebop_Ju1.mp3" controls></audio>| 




## Authors
Ryo Ogasawara

## References
