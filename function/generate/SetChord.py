import numpy as np
import copy

# コード進行のデータを作成、出力
class ChordProgressData(object):
    def __init__(self):
        self.SetKey = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3,
                       'E': 4, 'F': 5, 'Gb': 6, 'G': 7,
                       'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}

    def SetChord(self, Key, ChordType, Num=1):
        Output = self.ChordProgress(Key, ChordType, Num)
        return Output

    def ChordProgress(self, Key, ChordType, Num=1):
        Root1 = self.SetRoot(Key, 1)
        Root2 = self.SetRoot(Key, 2)
        Root3 = self.SetRoot(Key, 3)
        Root4 = self.SetRoot(Key, 4)
        Root5 = self.SetRoot(Key, 5)
        Root6 = self.SetRoot(Key, 6)
        Root7 = self.SetRoot(Key, 7)
        if ChordType == 'Blues1':
            OutputRoot = [Root1]*2+[Root4]*2+[Root1]*4 + [Root4]*4+[Root1]*2+[Root6]*2 + [Root2]*2+[Root5]*2+[Root1]*4
            OutputKind = ['dominant-seventh']*16+['minor']*2+['dominant-seventh']*6
            OutputRoot *= Num
            OutputKind *= Num
            Output = [OutputRoot, OutputKind]

        elif ChordType == 'Blues2':
            OutputRoot = [Root1]*2+[Root4]*2+[Root1]*4 +\
                         [Root4]*4+[Root1]*2 + [Root6]*2 + \
                         [Root2]*2+[Root5]*2+\
                         [Root1]*1+[Root6]*1+[Root2]*1+[Root5]*1
            OutputKind = ['dominant-seventh']*16+['minor']*2+['dominant-seventh']*4+['minor']*1+['dominant-seventh']*1
            OutputRoot *= Num
            OutputKind *= Num
            OutputRoot += [Root1]*2
            OutputKind += ['dominant-seventh']*2
            Output = [OutputRoot, OutputKind]

        elif ChordType == 'Two_Five_One':
            OutputRoot = [Root2]*2+[Root5]*2+[Root1]*4 + [Root2]*2+[Root5]*2+[Root1]*4 
            OutputKind = ['minor']*2+['major']*6 + ['minor']*2+['major']*6
            OutputRoot *= Num
            OutputKind *= Num
            Output = [OutputRoot, OutputKind]

        elif ChordType == 'Just_The_Two_Of_Us':
            OutputRoot = [Root4]*2+[Root3]*2+[Root6]*2+[Root5]*1+[Root1]*1 +  \
                         [Root4]*2+[Root3]*2+[Root6]*4
            OutputKind = ['major']*2+['dominant-seventh']*2+['minor']*3+['dominant-seventh']*1 +  \
                         ['major']*2+['dominant-seventh']*2+['minor']*4
            OutputRoot *= Num
            OutputKind *= Num
            Output = [OutputRoot, OutputKind]

        elif ChordType == 'Autumn_Leaves':
            OutputRoot = [Root2]*2+[Root5]*2+[Root1]*2+[Root4]*2+[Root7]*2+[Root3]*2 +[Root6]*4 
            OutputKind = ['minor']*2+['dominant-seventh']*2+['major']*4+\
                         ['half-diminished-seventh']*2+['dominant-seventh']*2+['minor']*4
            OutputRoot *= Num
            OutputKind *= Num
            Output = [OutputRoot, OutputKind]
        return Output

    def SetRoot(self, Key, Degree):
        ChordRoot = [0, 2, 4, 5, 7, 9, 11]
        KeyValue = self.SetKey[Key]
        return (ChordRoot[Degree-1]+KeyValue) % 12

def SetData(Note, Duration, OutStrChord):
    NoteInput = {'C': 72, 'Db': 73, 'D': 74, 'Eb': 75,
                 'E': 76, 'F': 77, 'Gb': 78, 'G': 79,
                 'Ab': 80, 'A': 81, 'Bb': 70, 'B': 71,
                 'Rest': 128}
    DurationInput = {'8': 240, '4': 480}
    OutNote = [NoteInput[Note]]
    OutDuration = [DurationInput[Duration]]
    return OutStrChord, OutNote, OutDuration

#　入力コード設定
class InputChord(object):
    def __init__(self):
        self.Chord = []
        self.ChordProgression = []
        self.measure = 0

    def SetChord(self, CHORD, StartChordNum=1, EndChordNum=1):
        for Root, Kind in zip(CHORD[0], CHORD[1]):
            self.Chord.append([Root, Kind])
        ChordRoot, ChordKind = copy.copy(CHORD[0]), copy.copy(CHORD[1])
        self.measure = int(len(ChordRoot) / 2)
        ChordRoot, ChordKind = ChordRoot+['<eos>','<eos>']*EndChordNum, ChordKind+['<eos>','<eos>']*EndChordNum
        ChordRoot, ChordKind = ['<s>','<s>']*StartChordNum+ChordRoot, ['<s>','<s>']*StartChordNum+ChordKind
        for idx in range(self.measure):
            IdxStart, IdxEnd = idx*2, idx*2+(StartChordNum+EndChordNum+1)*2
            self.ChordProgression.append(
                [ChordRoot[IdxStart:IdxEnd], ChordKind[IdxStart:IdxEnd]])
            self.ChordProgression.append(
                [ChordRoot[IdxStart:IdxEnd], ChordKind[IdxStart:IdxEnd]])

    def OutChord(self, TICK):
        assert TICK < self.measure*1920
        ChordIdx = int(TICK / 960)
        Chord = self.Chord[ChordIdx]
        ChordProgression = self.ChordProgression[ChordIdx]
        return Chord, ChordProgression

