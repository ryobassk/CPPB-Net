from anytree import Node
import torch
import torch.nn as nn
import music21 as m21
import pickle
import numpy as np
import random
import copy

#　データの選択
class DataSelect(object):
    def __init__(self, Corpus, Device, rulename='PitchRule1', 
                 startmidinum=0, endmidinum=127, Dataset2RuleDict_Path=None):
        self.corpus = Corpus
        self.device = Device
        self.rulename = rulename
        self.startmidinum = startmidinum
        self.endmidinum = endmidinum
        if rulename == 'PitchRule2' or rulename == 'PitchRule4' or rulename == 'PitchRule6':
            f = open(Dataset2RuleDict_Path, 'rb')
            self.DictPitch = pickle.load(f)

    def Select(self, OutPitch, OutDur, Offset, InDuration):
        PotentialPitch, PotentialDuration = torch.argsort(-OutPitch[-1]), torch.argsort(-OutDur[-1])
        PotentialPitch, PotentialDuration = self.id2word(PotentialPitch, PotentialDuration)
        NowOffset = Offset + InDuration
        #　音価の候補から選択
        RemainingOffset = 1920 - NowOffset % 1920
        for DurationIdx, Duration in enumerate(PotentialDuration):
            NextOffset = (NowOffset % 1920+Duration) % 1920
            if RemainingOffset >= Duration and NextOffset in self.corpus.DictOffset.word2idx:
                break
        PitchRule = getattr(self, self.rulename)
        PotentialPitchIdx = (np.array(range(0, len(PotentialPitch), 1))).tolist()
        PotentialPitch, PotentialPitchIdx = self.SetRangeInstrument(PotentialPitch, PotentialPitchIdx)
        Pitch, PitchIdx = PitchRule(PotentialPitch, PotentialPitchIdx, Duration, NowOffset)
        return Pitch, Duration, NowOffset, PitchIdx, DurationIdx,

    def id2word(self, PotentialPitch, PotentialDuration):
        WordPitch, WordDuration = [], []
        for Pitch in PotentialPitch:
            WordPitch.append(self.corpus.DictPitch.idx2word[Pitch])
        for duration in PotentialDuration:
            WordDuration.append(self.corpus.DictDuration.idx2word[duration])
        return WordPitch, WordDuration

    def SetChordDef(self, ChordDef):
        self.ChordDef = ChordDef

    #　楽器の音域設定
    def SetRangeInstrument(self, PotentialPitch, PotentialPitchIdx):
        PitchList = []
        PitchIdxList = []
        for Idx, Pitch in zip(PotentialPitchIdx, PotentialPitch):
            if type(Pitch) is str:
                continue
            if (Pitch >= self.startmidinum and Pitch <= self.endmidinum) or Pitch==128:
                PitchList.append(Pitch)
                PitchIdxList.append(Idx)
        return PitchList, PitchIdxList
    
    def SetChordTone(self, ChordRoot, ChordKind):
        if ChordKind == 'dominant-seventh':
            ChordTone = [1, 0, 0, 0,
                         1, 0, 0, 1,
                         0, 0, 1, 0]
        elif ChordKind == 'minor':
            ChordTone = [1, 0, 0, 1,
                         0, 0, 0, 1,
                         0, 0, 0, 0]
        elif ChordKind == 'major':
            ChordTone = [1, 0, 0, 0,
                         1, 0, 0, 1,
                         0, 0, 0, 0]
        elif ChordKind == 'half-diminished-seventh':
            ChordTone = [1, 0, 0, 1,
                         0, 0, 1, 0,
                         0, 0, 1, 0]
        ChordTone = (np.roll(ChordTone, ChordRoot)).tolist()
        ChordTone = self.ChordToken(ChordTone)
        return ChordTone

    def ChordToken(self, Data):  # コード音のみを抽出
        ChordList = np.array(Data)
        ChordList = np.nonzero(ChordList)[0]
        length = ChordList.size
        if length < 4:
            ChordList = np.append(ChordList, [12]*(4-length))
        Output = ChordList.tolist()
        return Output

    #　確率が１位を使う
    def PitchRule1(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=2):
        return PotentialPitch[0:1], PotentialPitchIdx[0:1]

    #　データセットにある音名を使用（コード、音価、オフセットの条件で）
    def PitchRule2(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        Chord = self.ChordDef.OutChord(NowOffset)[0]
        PitchClassList = self.DictPitch[Chord[0]][Chord[1]][NowOffset%1920][Duration]
        PitchList = []
        PitchIdxList = []
        for idx, Pitch in zip(PotentialPitchIdx, PotentialPitch):
            if type(Pitch) is str:
                continue
            if Pitch == 128:
                PitchClass = 12
            else:
                PitchClass = Pitch % 12
            if PitchClassList[PitchClass]!=0:
                PitchList.append(Pitch)
                PitchIdxList.append(idx)
            if idx >= num-1:
                break
        #print(PitchIdxList)
        return PitchList, PitchIdxList
    
    #　1拍目とコード変化時に３拍目をコードトーンにする
    def PitchRule3(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        if NowOffset % 1920 != 960 and NowOffset % 1920 != 0:
            return PotentialPitch, PotentialPitchIdx

        Chord = self.ChordDef.OutChord(NowOffset)[0]
        if NowOffset % 1920 == 960:
            PreChord = self.ChordDef.OutChord(NowOffset-960)[0]
            if PreChord[0]==Chord[0] and PreChord[1]==Chord[1]:
                return PotentialPitch, PotentialPitchIdx
        
        ChordTone = self.SetChordTone(Chord[0], Chord[1])
        PitchList = []
        PitchIdxList = []
        for idx, Pitch in zip(PotentialPitchIdx, PotentialPitch):
            if type(Pitch) is str:
                continue
            if Pitch == 128:
                PitchClass = 12
            else:
                PitchClass = Pitch % 12
            if PitchClass in ChordTone or PitchClass==12:
                PitchList.append(Pitch)
                PitchIdxList.append(idx)
            if idx >= num-1:
                break
        #print(PitchIdxList)
        return PitchList, PitchIdxList

    #　1拍目と３拍目をコードトーンにする
    def PitchRule5(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        if NowOffset % 1920 != 960 and NowOffset % 1920 != 0:
            return PotentialPitch, PotentialPitchIdx
        Chord = self.ChordDef.OutChord(NowOffset)[0]
        ChordTone = self.SetChordTone(Chord[0], Chord[1])
        PitchList = []
        PitchIdxList = []
        for idx, Pitch in zip(PotentialPitchIdx, PotentialPitch):
            if type(Pitch) is str:
                continue
            if Pitch == 128:
                PitchClass = 12
            else:
                PitchClass = Pitch % 12
            if PitchClass in ChordTone or PitchClass==12:
                PitchList.append(Pitch)
                PitchIdxList.append(idx)
            if idx >= num-1:
                break
        #print(PitchIdxList)
        return PitchList, PitchIdxList

    #　1拍目と３拍目をコードトーンにする と　データセットにある音名を使用（コード、音価、オフセットの条件で）
    def PitchRule4(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        PotentialPitch1, PotentialPitchIdx1 = self.PitchRule3(
            PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=num)
        PotentialPitch2, PotentialPitchIdx2 = self.PitchRule2(
            PotentialPitch1, PotentialPitchIdx1, Duration, NowOffset, num=num)
        return PotentialPitch2, PotentialPitchIdx2
    
    #　1拍目とコード変化時に３拍目をコードトーンにする と　データセットにある音名を使用（コード、音価、オフセットの条件で）
    def PitchRule6(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        PotentialPitch1, PotentialPitchIdx1 = self.PitchRule5(
            PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=num)
        PotentialPitch2, PotentialPitchIdx2 = self.PitchRule2(
            PotentialPitch1, PotentialPitchIdx1, Duration, NowOffset, num=num)
        return PotentialPitch2, PotentialPitchIdx2


#　ランダムデータの選択
class RandomDataSelect(object):
    def __init__(self, Corpus, Device, rulename='PitchRule1', 
                 startmidinum=0, endmidinum=127, Dataset2RuleDict_Path=None):
        self.corpus = Corpus
        self.device = Device
        self.rulename = rulename
        self.startmidinum = startmidinum
        self.endmidinum = endmidinum
        self.Softmax = nn.Softmax()
        if rulename == 'PitchRule2' or rulename == 'PitchRule4' or rulename == 'PitchRule6':
            f = open(Dataset2RuleDict_Path, 'rb')
            self.DictPitch = pickle.load(f)
        
        

    def Select(self, OutPitch, OutDur, Offset, InDuration):
        PotentialPitch, PotentialDuration = torch.argsort(-OutPitch[-1]), torch.argsort(-OutDur[-1])
        PotentialPitch_value, PotentialDuration_value = torch.sort(
            OutPitch[-1], descending=True).values, torch.sort(OutDur[-1], descending=True).values
        PotentialPitch, PotentialDuration = self.id2word(PotentialPitch, PotentialDuration)
        NowOffset = Offset + InDuration
        #　音価の候補から選択
        Duration, DurationIdx = self.DurationRandomchoice(PotentialDuration, PotentialDuration_value, NowOffset)
        if Duration==None:return [],[],[],[],[]
        PitchRule = getattr(self, self.rulename)
        PotentialPitchIdx = (np.array(range(0, len(PotentialPitch), 1))).tolist()
        PotentialPitch, PotentialPitchIdx = self.SetRangeInstrument(PotentialPitch, PotentialPitchIdx)
        
        if self.rulename == 'PitchRule1':
            Pitch, PitchIdx = PotentialPitch, PotentialPitchIdx
            RuleFlag = True
        else:
            Pitch, PitchIdx = PitchRule(PotentialPitch, PotentialPitchIdx,
                                        Duration, NowOffset)
            RuleFlag = False
        Pitch, PitchIdx = self.PitchRandomchoice(Pitch, PitchIdx, PotentialPitch_value, RuleFlag=RuleFlag)
        return Pitch, Duration, NowOffset, PitchIdx, DurationIdx,

    def PitchRandomchoice(self, Pitch, PitchIdx, PotentialPitch_value, RuleFlag=False):
        RandomPitch, RandomPitchIdx = [], []
        random_weight = []
        PitchIdxDict = {}
        for pitch, idx in zip(Pitch, PitchIdx):
            PitchIdxDict[pitch] = idx
        for idx in PitchIdx:
            random_weight.append(PotentialPitch_value[idx].item())
        random_weight = self.Softmax(torch.tensor(random_weight))
        random_weight = random_weight.tolist()
        random_weight_copy = copy.deepcopy(random_weight)
        if RuleFlag:
            indices = range(len(Pitch))
            for i in range(1):
                idx = random.choices(indices, random_weight_copy)[0]
                RandomPitch.append(Pitch[idx])
                RandomPitchIdx.append(PitchIdxDict[Pitch[idx]])
        else:
            indices = range(len(Pitch))
            for i in range(len(Pitch)):
                idx = random.choices(indices, random_weight_copy)[0]
                RandomPitch.append(Pitch[idx])
                RandomPitchIdx.append(PitchIdxDict[Pitch[idx]])
                random_weight_copy[idx] = 0
        return RandomPitch, RandomPitchIdx

    def DurationRandomchoice(self, PotentialDuration, PotentialDuration_value, NowOffset):
        RandomDuration, RandomDurationIdx = [], []
        random_weight = self.Softmax(torch.tensor(PotentialDuration_value[0:5]))
        random_weight = random_weight.tolist()
        PotentialDuration_copy = copy.deepcopy(PotentialDuration[0:5])
        random_weight_copy = copy.deepcopy(random_weight)
        indices = range(len(PotentialDuration_copy))
        for i in range(len(PotentialDuration_copy)):
            idx = random.choices(indices, random_weight_copy)[0]
            RandomDuration.append(PotentialDuration_copy[idx])
            RandomDurationIdx.append(idx)
            random_weight_copy[idx] = 0

        OutDuration, OutDurationIdx = [], []
        RemainingOffset = 1920 - NowOffset % 1920
        for Duration, DurationIdx in zip(RandomDuration, RandomDurationIdx):
            if type(Duration) is str:
                continue
            NextOffset = (NowOffset % 1920+Duration) % 1920
            if RemainingOffset >= Duration and NextOffset in self.corpus.DictOffset.word2idx:
                OutDuration.append(Duration)
                OutDurationIdx.append(DurationIdx)
        if len(OutDuration)==0: return None, None
        return OutDuration[0], OutDurationIdx[0]

    def id2word(self, PotentialPitch, PotentialDuration):
        WordPitch, WordDuration = [], []
        for Pitch in PotentialPitch:
            WordPitch.append(self.corpus.DictPitch.idx2word[Pitch])
        for duration in PotentialDuration:
            WordDuration.append(self.corpus.DictDuration.idx2word[duration])
        return WordPitch, WordDuration

    def SetChordDef(self, ChordDef):
        self.ChordDef = ChordDef

    #　楽器の音域設定
    def SetRangeInstrument(self, PotentialPitch, PotentialPitchIdx):
        PitchList = []
        PitchIdxList = []
        for Idx, Pitch in zip(PotentialPitchIdx, PotentialPitch):
            if type(Pitch) is str:
                continue
            if (Pitch >= self.startmidinum and Pitch <= self.endmidinum) or Pitch==128:
                PitchList.append(Pitch)
                PitchIdxList.append(Idx)
        return PitchList, PitchIdxList
    
    def SetChordTone(self, ChordRoot, ChordKind):
        if ChordKind == 'dominant-seventh':
            ChordTone = [1, 0, 0, 0,
                         1, 0, 0, 1,
                         0, 0, 1, 0]
        elif ChordKind == 'minor':
            ChordTone = [1, 0, 0, 1,
                         0, 0, 0, 1,
                         0, 0, 0, 0]
        elif ChordKind == 'major':
            ChordTone = [1, 0, 0, 0,
                         1, 0, 0, 1,
                         0, 0, 0, 0]
        elif ChordKind == 'half-diminished-seventh':
            ChordTone = [1, 0, 0, 1,
                         0, 0, 1, 0,
                         0, 0, 1, 0]
        ChordTone = (np.roll(ChordTone, ChordRoot)).tolist()
        ChordTone = self.ChordToken(ChordTone)
        return ChordTone

    def ChordToken(self, Data):  # コード音のみを抽出
        ChordList = np.array(Data)
        ChordList = np.nonzero(ChordList)[0]
        length = ChordList.size
        if length < 4:
            ChordList = np.append(ChordList, [12]*(4-length))
        Output = ChordList.tolist()
        return Output

    #　確率が１位を使う
    def PitchRule1(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=2):
        return PotentialPitch[0:1], PotentialPitchIdx[0:1]

    #　データセットにある音名を使用（コード、音価、オフセットの条件で）
    def PitchRule2(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        Chord = self.ChordDef.OutChord(NowOffset)[0]
        PitchClassList = self.DictPitch[Chord[0]][Chord[1]][NowOffset%1920][Duration]
        PitchList = []
        PitchIdxList = []
        for idx, Pitch in zip(PotentialPitchIdx, PotentialPitch):
            if type(Pitch) is str:
                continue
            if Pitch == 128:
                PitchClass = 12
            else:
                PitchClass = Pitch % 12
            if PitchClassList[PitchClass]!=0:
                PitchList.append(Pitch)
                PitchIdxList.append(idx)
            if idx >= num-1:
                break
        #print(PitchIdxList)
        return PitchList, PitchIdxList
    
    #　1拍目とコード変化時に３拍目をコードトーンにする
    def PitchRule3(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        if NowOffset % 1920 != 960 and NowOffset % 1920 != 0:
            return PotentialPitch, PotentialPitchIdx

        Chord = self.ChordDef.OutChord(NowOffset)[0]
        if NowOffset % 1920 == 960:
            PreChord = self.ChordDef.OutChord(NowOffset-960)[0]
            if PreChord[0]==Chord[0] and PreChord[1]==Chord[1]:
                return PotentialPitch, PotentialPitchIdx
        
        ChordTone = self.SetChordTone(Chord[0], Chord[1])
        PitchList = []
        PitchIdxList = []
        for idx, Pitch in zip(PotentialPitchIdx, PotentialPitch):
            if type(Pitch) is str:
                continue
            if Pitch == 128:
                PitchClass = 12
            else:
                PitchClass = Pitch % 12
            if PitchClass in ChordTone or PitchClass==12:
                PitchList.append(Pitch)
                PitchIdxList.append(idx)
            if idx >= num-1:
                break
        #print(PitchIdxList)
        return PitchList, PitchIdxList

    #　1拍目と３拍目をコードトーンにする
    def PitchRule5(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        if NowOffset % 1920 != 960 and NowOffset % 1920 != 0:
            return PotentialPitch, PotentialPitchIdx
        Chord = self.ChordDef.OutChord(NowOffset)[0]
        ChordTone = self.SetChordTone(Chord[0], Chord[1])
        PitchList = []
        PitchIdxList = []
        for idx, Pitch in zip(PotentialPitchIdx, PotentialPitch):
            if type(Pitch) is str:
                continue
            if Pitch == 128:
                PitchClass = 12
            else:
                PitchClass = Pitch % 12
            if PitchClass in ChordTone or PitchClass==12:
                PitchList.append(Pitch)
                PitchIdxList.append(idx)
            if idx >= num-1:
                break
        #print(PitchIdxList)
        return PitchList, PitchIdxList

    #　1拍目と３拍目をコードトーンにする と　データセットにある音名を使用（コード、音価、オフセットの条件で）
    def PitchRule4(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        PotentialPitch1, PotentialPitchIdx1 = self.PitchRule3(
            PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=num)
        PotentialPitch2, PotentialPitchIdx2 = self.PitchRule2(
            PotentialPitch1, PotentialPitchIdx1, Duration, NowOffset, num=num)
        return PotentialPitch2, PotentialPitchIdx2
    
    #　1拍目とコード変化時に３拍目をコードトーンにする と　データセットにある音名を使用（コード、音価、オフセットの条件で）
    def PitchRule6(self, PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=5):
        PotentialPitch1, PotentialPitchIdx1 = self.PitchRule5(
            PotentialPitch, PotentialPitchIdx, Duration, NowOffset, num=num)
        PotentialPitch2, PotentialPitchIdx2 = self.PitchRule2(
            PotentialPitch1, PotentialPitchIdx1, Duration, NowOffset, num=num)
        return PotentialPitch2, PotentialPitchIdx2

































#　データの変換
class Data2Tensor(object):
    def __init__(self, Corpus, Device):
        self.corpus = Corpus
        self.device = Device

    def converter(self, STRCHORD, CHORD,  PITCH, DURATION, TICK):
        OFFSET = TICK % 1920
        ChordRoot, ChordKind = self.corpus.DictChordRoot.word2idx[CHORD[0]], self.corpus.DictChordKind.word2idx[CHORD[1]]
        Pitch = self.corpus.DictPitch.word2idx[PITCH]
        Duration = self.corpus.DictDuration.word2idx[DURATION]
        Offset = self.corpus.DictOffset.word2idx[OFFSET]
        Data = [ChordRoot, ChordKind,
                Pitch, Duration, Offset]
        Data = self.Data2Batch(Data)
        Data = torch.tensor(Data).to(self.device)
        StrChord = self.StrChord2Batch(STRCHORD)
        return Data, StrChord

    def Data2Batch(self, Data):
        Batch = []
        for data in Data:
            Batch.append([[data]])
        return Batch

    def StrChord2Batch(self, StrChord):
        Batch = []
        IdStrChord = []
        StrChordRoot, StrChordKind = [], []
        for Chord in StrChord[0]:
            StrChordRoot.append(self.corpus.DictChordRoot.word2idx[Chord])
        for Chord in StrChord[1]:
            StrChordKind.append(self.corpus.DictChordKind.word2idx[Chord])
        IdStrChord = [[[StrChordRoot]], [[StrChordKind]]]
        IdStrChord = torch.tensor(IdStrChord).to(self.device)
        return IdStrChord


#　データの保存 xmlの出力
class DataSave(object):
    def __init__(self, Type):
        self.Chord = []
        self.Pitch = []
        self.Duration = []
        self.Type = Type
        self.ChordClass2Key = ['C', 'D-', 'D', 'E-',
                               'E', 'F', 'G-', 'G', 'A-', 'A', 'B-', 'B']
        self.ChordKind2Kind = {'dominant-seventh': '7', 'minor': 'm',
                               'major': '', 'half-diminished-seventh': 'm7b5'}

    def SetChord(self, ChordList):
        for ChordRoot, ChordKind in zip(ChordList[0], ChordList[1]):
            self.Chord.append(self.Chord2Figure([ChordRoot, ChordKind]))

    def Chord2Figure(self, Chord):
        ChordRoot = self.ChordClass2Key[Chord[0]]
        ChordKind = self.ChordKind2Kind[Chord[1]]
        ChordFigure = ChordRoot+ChordKind
        return ChordFigure

    def SetNote(self, PitchList, DurationList):
        self.Pitch = PitchList
        self.Duration = DurationList

    def Output(self, Name, Path, Type='xml', Tempo=160, Musescore=False):
        GenerateData = m21.stream.Stream()  # 楽譜オブジェクトの生成
        GenerateData.append(m21.clef.TrebleClef())  # ト音記号に設定
        GenerateData.append(m21.meter.TimeSignature('4/4'))  # 四分の四に設定
        GenerateData.append(m21.tempo.MetronomeMark(number=Tempo))
        Offset = 0.0
        flag = 0
        for pitch, duration in zip(self.Pitch, self.Duration):
            # Tick を　music21用に変換　（4分音符が1)
            quarterLength = duration / 480.0
            OffsetLength = Offset / 480.0
            ChordIdx = int(Offset/960)
            if (Offset % 1920 == 0) or (Offset % 1920 == 960 and self.Chord[ChordIdx] != self.Chord[ChordIdx-1]):
                ChordSymbol = m21.harmony.ChordSymbol()
                ChordIdx = int(Offset/960)
                ChordSymbol.figure = self.Chord[ChordIdx]
                GenerateData.append(ChordSymbol)
            if pitch == 128:  # 音階情報追加（休符時)
                RestSymbol = m21.note.Rest(
                    quarterLength=quarterLength, offset=OffsetLength)
                GenerateData.append(RestSymbol)
            else:  # 音階情報追加（音符時)
                NoteSymbol = m21.note.Note(pitch, quarterLength=quarterLength, offset=OffsetLength)
                GenerateData.append(NoteSymbol)
            Offset += duration
            flag += 1
        GenerateData.makeMeasures(inPlace=True)  # 小節追加
        if Musescore:  # 楽譜をmusescoreで表示する(pathを設定する必要あり)
            GenerateData.show('musicxml', addEndTimes=True)
        if Type == 'xml':  # Xmlファイル保存
            GenerateData.write('musicxml', '{}/{}.xml'.format(Path, Name))
        elif Type == 'mid':  # MIDIファイルで保存
            GenerateData.write('midi', '{}/{}.mid'.format(Path, Name))
