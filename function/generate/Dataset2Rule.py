import pickle
import pandas as pd
import os
import glob
import copy
import numpy as np

#　Csvファイルのデータを取得
def ExtractCsv(Path, Artist=None):
    Output = []
    LoadData = pd.read_csv(Path, header=0, index_col=0)
    LoadData = LoadData.values.tolist()
    Key, Name = (os.path.splitext(os.path.basename(Path))[0]).split('~')
    Key = (Key.split('_'))[0]
    for Data in LoadData:
        ChordClass, ChordKind = Data[4], Data[5]
        Duration, Offset = Data[1], Data[2]
        if Data[0]!=128:
            NoteKind = 'Note'
            Midi, PitchClass, Octave = Data[0], Data[0]%12, int(Data[0]/12)
        else:
            NoteKind = 'Rest'
            Midi, PitchClass, Octave = 128, 128, 128
        Note = [Artist, Name, Key, NoteKind,
                ChordClass, ChordKind,
                Midi, PitchClass, Octave,
                Duration, Offset]
        Output.append(Note)
    return Output

#　Csvデータの読み込みと出力
def CsvLoadOutput(Path, Artist=None):
    DataList = sorted(glob.glob('{}/*.csv'.format(Path)))
    if DataList == []:print('ファイルが存在しません')
    LoadData = []
    for DataPath in DataList:
        Data = ExtractCsv(DataPath, Artist=Artist)
        LoadData.append(Data)
    return LoadData

#　データを曲ごとに分けているのをなくす
def ExpandData(LoadData):
    Output = []
    for MusicData in LoadData:
        for Data in MusicData:
            Output.append(Data)
    return Output

# データセットからルールを作成
# コード、音価、Offsetの条件時の音名使用辞書を作成
class DataDict():
    def __init__(self):
        self.DictNote = {}
    def loaddata(self, LoadData):
        OffsetList = []
        DurationList = []
        ChordRootList = []
        ChordKindList = []
        for data in LoadData:
            chordroot, chordkind = data[4], data[5]
            duration, offset = data[9], data[10]
            ChordRootList.append(chordroot)
            ChordKindList.append(chordkind)
            DurationList.append(duration)
            OffsetList.append(offset)
        ChordRootList = sorted(list(set(ChordRootList)))
        ChordKindList = sorted(list(set(ChordKindList)))
        OffsetList = sorted(list(set(OffsetList)))
        DurationList = sorted(list(set(DurationList)))
        ChordRootDict = {}
        for chordroot in ChordRootList:
            ChordRootData = [Data for Data in LoadData if Data[4] == chordroot]
            ChordKindDict = {}
            for chordkind in ChordKindList:
                ChordKindData = [Data for Data in ChordRootData if Data[5] == chordkind]
                OffsetDict = {}
                for offset in OffsetList:
                    OffsetData = [Data for Data in ChordKindData if Data[10] == offset]
                    DurationDict = {}
                    for duration in DurationList:
                        DurationData = [Data for Data in OffsetData if Data[9] == duration]
                        pitchclasslist = [0]*13
                        for data in DurationData:
                            pitchclass = data[7]
                            if pitchclass == 128:
                                pitchclass = 12
                            pitchclasslist[pitchclass] += 1
                        DurationDict[duration] = copy.deepcopy(pitchclasslist)
                    OffsetDict[offset] = copy.deepcopy(DurationDict)
                ChordKindDict[chordkind] = copy.deepcopy(OffsetDict)
            self.DictNote[chordroot] = copy.deepcopy(ChordKindDict)
            ChordRootDict[chordroot] = copy.deepcopy(ChordKindDict)
