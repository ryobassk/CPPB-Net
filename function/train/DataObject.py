import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

class Dictionary(object): # 文字とIDの辞書
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word): # 辞書に情報を追加
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self): # 辞書の大きさ
        return len(self.idx2word)

class AllDictionary(object):  # 使用するデータ表現の辞書
    def __init__(self):
        self.DictChordRoot = Dictionary()
        self.DictChordKind = Dictionary()
        self.DictPitch = Dictionary()
        self.DictDuration = Dictionary()
        self.DictOffset = Dictionary()

class Corpus(object):
    def __init__(self, path):
        self.Dict = AllDictionary() # 辞書の設定
        # 特殊文字の辞書追加（'<pad>':パッディング文字, '<s>':開始記号, '<eos>':終了記号, '<unk>':未知文字）
        for word_spe in ['<pad>', '<s>', '<eos>', '<unk>']:
            self.Dict.DictChordRoot.add_word(word_spe)
            self.Dict.DictChordKind.add_word(word_spe)
            self.Dict.DictPitch.add_word(word_spe)
            self.Dict.DictDuration.add_word(word_spe)
            self.Dict.DictOffset.add_word(word_spe)
        Data = self.data_load(path)                                                     # データの読み込み
        Dataset = self.tokenize(Data)                                                   # データのID変換
        self.train, self.test = train_test_split(Dataset, test_size=0.1, shuffle=True)  # データを分割

    def data_load(self, path): # csvデータの読み込み
        # ファイルパスの存在を確認
        print(path)
        assert os.path.exists(path+'/note/') or os.path.exists(path+'/chord/')
        file_csv = glob.glob(path+'/note/*.csv') # パスの取得
        Data = []                                # データの保存先（各曲ずつ情報を収納）
        for path_csv in file_csv:  # csvの読み込み
            #   音符情報
            DataFrame = pd.read_csv(path_csv, header=0, index_col=0) 
            ChordRoot = DataFrame['ChordRoot'].values.tolist()        # コードのルート(音符情報)
            ChordKind = DataFrame['ChordKind'].values.tolist()        # コードの種類(音符情報)
            Pitch = DataFrame['Pitch'].values.tolist()                # 音名
            Duration = DataFrame['Duration'].values.tolist()          # 音価
            Offset = DataFrame['Offset'].values.tolist()              # offset
            Measure = DataFrame['Measure'].values.tolist()            # 小節数
            # コード進行情報
            path_chord = '{}/chord/{}'.format(path, os.path.basename(path_csv))
            DataFrame = pd.read_csv(path_chord, header=0, index_col=0)
            StrChordRoot = DataFrame['ChordClass'].values.tolist()    # コードのルート(コード進行情報)
            StrChordKind = DataFrame['ChordKind'].values.tolist()     # コードの種類(コード進行情報)
            # 0:コードのルート(コード進行情報), 1:コードの種類(コード進行情報), 
            # 2:コードのルート(音符情報), 3:コードの種類(音符情報), 
            # 4:音名, 5:音価, 6:offset, 7:小節数,
            Data.append([StrChordRoot, StrChordKind, ChordRoot, ChordKind, Pitch, Duration, Offset, Measure])
        return Data

    def tokenize(self, Data):
        # 辞書の作成
        for musicdata in Data:
            for idx, words in enumerate(musicdata):
                # words　↓
                # 0:コードのルート(コード進行情報), 1:コードの種類(コード進行情報),
                # 2:コードのルート(音符情報), 3:コードの種類(音符情報),
                # 4:音名, 5:音価, 6:offset, 7:小節数,
                for word in words:
                    if idx == 0:   self.Dict.DictChordRoot.add_word(word)
                    elif idx == 1: self.Dict.DictChordKind.add_word(word)
                    elif idx == 2: continue
                    elif idx == 3: continue
                    elif idx == 4: self.Dict.DictPitch.add_word(word)
                    elif idx == 5: self.Dict.DictDuration.add_word(word)
                    elif idx == 6: self.Dict.DictOffset.add_word(word)
                    elif idx == 7: continue
        # データセットの作成
        Data = self.Data2Id(Data)             # データのID変換
        measure_idx = self.Measure2Idx(Data)  # 1小節の区切りとidxの対応関係を取得
        windowLen = 8                         # 区切る小節数
        DatasetInfo = self.WindowData(Data, measure_idx, windowLen)  # window幅のデータセット作成
        return DatasetInfo

    def Data2Id(self, Data):  # データの文字を IDに変換
        OutData = []
        for musicdata in Data:
            StrchordRoot, StrchordKind, chordRoot, chordKind = [], [], [], []
            note, duration, offset= [], [], []
            for idx, separatedata in enumerate(musicdata):
                for word in separatedata:
                    if idx == 0:StrchordRoot.append(self.Dict.DictChordRoot.word2idx[word])
                    elif idx == 1:StrchordKind.append(self.Dict.DictChordKind.word2idx[word])
                    elif idx == 2:chordRoot.append(self.Dict.DictChordRoot.word2idx[word])
                    elif idx == 3:chordKind.append(self.Dict.DictChordKind.word2idx[word])
                    elif idx == 4:note.append(self.Dict.DictPitch.word2idx[word])
                    elif idx == 5:duration.append(self.Dict.DictDuration.word2idx[word])
                    elif idx == 6:offset.append(self.Dict.DictOffset.word2idx[word])
                    elif idx == 7: continue
            # 0:コードのルート(コード進行情報), 1:コードの種類(コード進行情報),
            # 2:コードのルート(音符情報), 3:コードの種類(音符情報),
            # 4:音名, 5:音価, 6:offset, 7:小節数,
            OutData.append([StrchordRoot, StrchordKind, chordRoot, chordKind, note, duration, offset, musicdata[7]])
        return OutData

    def Measure2Idx(self, data):  
        # 1 小節単位の区切り と 音符情報のidxとの 対応関係を 取得
        measure_num = []
        for musicdata in data:
            ids = []
            Offset = musicdata[6]
            for idx, offsetData in enumerate(Offset):
                if offsetData == self.Dict.DictOffset.word2idx[0.0]:
                    ids.append(idx)
            measure_num.append(ids)
        return measure_num

    def WindowData(self, Data, MeasureIdxDict, windowLen):  
        # window幅 の データセット 作成
        DatasetInfo = []  # window幅データセット の保存先
        for idxMusic, musicdata in enumerate(Data):
            idxMeasure = 0  # 切り出す先頭　の　小節数
            while len(MeasureIdxDict[idxMusic]) >= idxMeasure + windowLen: # 曲の終わりか　確認
                # Chord
                # 0:コードのルート(コード進行情報), 1:コードの種類(コード進行情報),
                # Phrase
                # 0:コードのルート(音符情報), 1:コードの種類(音符情報),
                # 2:音名, 3:音価, 4:offset, 5:小節数,
                Chord, Phrase  = musicdata[0:2], musicdata[2:]
                noteidxlen = len(Phrase[1])                                               # 音符情報のidxの長さ
                WinPhrase = self.WindowPhrase(Phrase, MeasureIdxDict, 
                                              idxMusic, idxMeasure, windowLen)            # 音符情報を作成
                WinChord = self.WindowChord(Chord, MeasureIdxDict, 
                                            idxMusic, idxMeasure, windowLen, noteidxlen)  # コード進行情報を作成
                idxMeasure += 1

                # WinPhraseの小節数を曲の絶対数ではなく、window幅内での相対数に変換 (WinPhraseの先頭のデータの小節数を０として考える)
                slide_n = WinPhrase[5][0]
                WinPhrase[5] = [n - slide_n for n in WinPhrase[5]]

                DatasetInfo.append([WinChord, WinPhrase]) #コード進行情報と音符情報を格納
        return DatasetInfo

    def WindowPhrase(self, notedata, MeasureIdxDict, idxMusic, idxMeasure, windowLen):  
        # 音符情報を  window幅で分割
        MeasureStartIdx = MeasureIdxDict[idxMusic][idxMeasure]  # 切り出す先頭の小節に対応した音符情報のindex
        MusicInfo = []  # window幅のフレーズ 保存先
        if len(MeasureIdxDict[idxMusic]) > idxMeasure + windowLen:  # window幅以内であるか
            # 切り出す最後の小節に対応した音符情報のindex
            MeasureEndIdx = MeasureIdxDict[idxMusic][idxMeasure + windowLen]
            for idx_data in range(len(notedata)):
                MusicInfo.append(notedata[idx_data][MeasureStartIdx:MeasureEndIdx])  # window幅のフレーズ
        else:                                                       # 音符情報の末尾のidxのとき
            for idx_data in range(len(notedata)):
                MusicInfo.append(notedata[idx_data][MeasureStartIdx:])               # window幅のフレーズ
        return MusicInfo

    def WindowChord(self, chorddata, MeasureIdxDict, idxMusic, idxMeasure, windowLen, noteidxlen):
        # コード進行情報を  window幅で分割

        # 取得する小節分のコード進行を取得
        SectionChord = self.ChordSection(chorddata, idxMeasure, windowLen)

        # 各コードに存在する音符の数のリスト
        SectionNoteNum = []
        prenum = 0
        for num in MeasureIdxDict[idxMusic][1:]:
            SectionNoteNum.append(num - prenum)
            prenum = num
        SectionNoteNum.append(noteidxlen - MeasureIdxDict[idxMusic][-1])

        # 使用する小節数の範囲の各コードに存在する音符数のリスト
        SectionNoteNum = SectionNoteNum[idxMeasure:idxMeasure+windowLen]
        # 各音符情報ごとのコード進行情報の取得
        StrChord = self.ChordFitNote(SectionChord, SectionNoteNum, windowLen)
        return StrChord
    
    def ChordSection(self, chorddata, idxMeasure, windowLen):  
        # 取得する小節分のコード進行を取得（この際に開始記号と終了記号を付与）
        StartMea = idxMeasure * 2              # 取得する最初の小節数
        EndMea = (idxMeasure + windowLen) * 2  # 取得する最後の小節数
        StartPadding = [1]                     # 開始記号
        EndPadding = [2]                       # 終了記号
        if idxMeasure == 0 and len(chorddata[0]) == EndMea:  
            # window幅で曲が始まり、終わるとき
            # 開始記号 + 取得する小節分のコード + 終了記号
            StrChord = [StartPadding*2 + chorddata[0][StartMea:EndMea] + EndPadding*2,
                        StartPadding*2 + chorddata[1][StartMea:EndMea] + EndPadding*2]

        elif idxMeasure == 0:                                
            # window幅で曲が始まるとき
            # 開始記号 + 取得する小節分のコード + 取得する小節分のコードの次の小節のコード
            StrChord = [StartPadding*2 + chorddata[0][StartMea:EndMea + 2],
                        StartPadding*2 + chorddata[1][StartMea:EndMea + 2]]

        elif len(chorddata[0]) == EndMea:                    
            # window幅で曲が終わるとき
            # 取得する小節分のコードの前の小節のコード + 取得する小節分のコード + 終了記号
            StrChord = [chorddata[0][StartMea - 2: EndMea] + EndPadding*2,
                        chorddata[1][StartMea - 2: EndMea] + EndPadding*2]

        else:                                                
            # window幅が曲の途中にあるとき
            # 取得する小節分のコードの前の小節のコード + 取得する小節分のコード + 取得する小節分のコードの次の小節のコード
            StrChord = [chorddata[0][StartMea - 2: EndMea + 2], chorddata[1][StartMea - 2: EndMea + 2]]
        return StrChord

    def ChordFitNote(self, SectionChord, SectionNoteNum, windowLen):  
        # 各音符情報ごとのコード進行情報の取得
        StrChord = []                # 音符情報に対応した　window幅コード進行情報の 保存先
        IndivSectionChordRoot = []   # 各小節ごとのコード進行情報（コードのルート）
        IndivSectionChordKind = []   # 各小節ごとのコード進行情報（コードの種類）
        ChordwindowLen = 3 *2        # 取得するコード進行の小節数（３小節）idxは0.5拍単位なので2倍する

        # 各小節ごとのコード進行情報を取得
        # (3 小節分のコード進行：現在の小節、前後の小節のコード)
        startidx = 0 
        for idx in range(windowLen): 
            IndivSectionChordRoot.append(SectionChord[0][startidx:startidx+ChordwindowLen])
            IndivSectionChordKind.append(SectionChord[1][startidx:startidx+ChordwindowLen])
            startidx += 2
        
        # 音符情報にコード進行情報を対応させる
        # 各音符ごとのコード進行情報に変形
        for chordnotenum, SecChordRoot, SecChordKind, in zip(SectionNoteNum, IndivSectionChordRoot,  IndivSectionChordKind):
            for idx in range(chordnotenum):  # 各小節の音符数(chordnotenum)分追加
                StrChord.append([SecChordRoot, SecChordKind])
        return StrChord
