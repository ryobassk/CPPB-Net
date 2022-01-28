from music21 import *
import pandas as pd
import numpy as np
import copy

#　Note　CSV　出力
def SaveData(Data, path, Type='Note'):
    if Type == 'Note':
        df = pd.DataFrame(Data, index=None, columns=['Pitch', 'Duration', 'Offset', 'Measure',
                                                    'ChordRoot', 'ChordKind',
                                                    'C', 'Db', 'D', 'Eb', 'E', 'F',
                                                    'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])
        df.to_csv(path)
    if Type == 'Chord':
        df = pd.DataFrame(Data, index=None, columns=['ChordClass', 'ChordKind', 
                                                     'Duration', 'Offset', 'Mesure',
                                                     'C', 'Db', 'D', 'Eb', 'E', 'F',
                                                     'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])
        df.to_csv(path)

#　xmlファイルのデータ取得
def XmlLoad(path):
    # xmlファイル読み込み
    MusicData = converter.parse(path)  
    # 楽曲のキー取得
    KeyInfo = MusicData.analyze('key')
    if KeyInfo.mode != "minor":Key = KeyInfo.tonic.name
    else:Key = KeyInfo.relative.tonic.name  
    # 情報(音符、コード)の保存先
    Notes, Chords = [], []
    for thisNote in MusicData.flat.notesAndRests:
        if thisNote.isChord: #コード時
            NowChord = Chord2ClassList(thisNote.orderedPitchClasses) # コードトーン
            NowChordClass = thisNote.pitchClasses[0]                 # コードのルート
            NowChordKind = thisNote.chordKind                        # コードの種類
            ChordDuration = thisNote.duration.quarterLength * 480    # 音価
            ChordOffset = thisNote.offset * 480                      # offset
            ChordMesure = thisNote.measureNumber                     # 小節数
            # 0:コードのルート, 1:コードの種類, 2,:音価 3:Offset, 4:小節数, 5〜16:コードトーン
            Chords.append([NowChordClass, NowChordKind, ChordDuration, ChordOffset, ChordMesure] + NowChord)
        else:  # 音符 or 休符時
            #　音名（MIDI Num : 音符は0〜127, 休符は128）
            if thisNote.isNote: NotePitch = thisNote.pitch.midi
            elif thisNote.isRest: NotePitch = 128
            NoteDuraion = thisNote.duration.quarterLength * 480  # 音価
            NoteOffset = thisNote.offset * 480 % 1920            # offset
            NoteMesure = thisNote.measureNumber                  # 小節数
            #0:音名, 1:音価, 2:offset, 3:小節数, 4:コードのルート, 5:コードの種類, 6〜17:コードトーン
            NoteInfo = [NotePitch, NoteDuraion, NoteOffset,
                        NoteMesure, NowChordClass, NowChordKind] + NowChord
            Notes.append(NoteInfo)
    Chords = ChordInfoConvert(Chords)  # コード情報を２拍ごとに分割
    return Notes, Chords, Key

# コード情報を２拍ごとに分割
def ChordInfoConvert(Chords):  
    Output = []                 # コード情報の保存先
    idxlist = []                # 2拍ごとに分割した際のChordsに対応したIdx
    NowOffset = 0               # 現在のoffset
    MaxOffset = Chords[-1][3]   # offsetの最大値
    # 2拍ごとに分割した際のChordsのIdxの取得
    while NowOffset <= MaxOffset:
        flag1, flag2 = 0, 0
        for idx, Chord in enumerate(Chords):
            Offset = Chord[3]
            if Offset > NowOffset:
                flag2 = 1
            if flag1 == 1 and flag2 == 1:
                break
            if Offset <= NowOffset:
                preidx = idx
                flag1 = 1
        NowOffset += 960
        idxlist.append(preidx)
        if NowOffset > MaxOffset and NowOffset % 1920 == 960:
            idxlist.append(preidx)
    # 2拍ごとに分割したChordsの作成
    NowOffset = 0
    for idx in idxlist:
        NowChord = Chords[idx]
        ChordClass = Chords[idx][0]
        ChordKind = Chords[idx][1]
        ChordDuration = 960
        ChordOffset = NowOffset % 1920
        ChordMesure = Chords[idx][4]
        NowChord = Chords[idx][5:]
        # 0:コードのルート, 1:コードの種類, 2,:音価 3:Offset, 4:小節数, 5〜16:コードトーン
        Output.append([ChordClass, ChordKind, ChordDuration,
                       ChordOffset, ChordMesure] + NowChord)
        NowOffset += 960
    return Output


# コードトーンを音名リストに変換
# [2, 5, 8, 10]　→　[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
def Chord2ClassList(Chords):
    ChordToneList = [0]*12
    for PitchClass in Chords:
        ChordToneList[PitchClass] = 1
    return ChordToneList

#　音符情報を　Value分　転調
def NoteModulation(NoteData, Value):
    Output = []  # 転調した音符情報の保存先
    if Value == 0:
        return NoteData
    for thisNote in NoteData:
        #NoteInfo↓
        # 0:音名, 1:音価, 2:offset, 3:小節数, 4:コードのルート, 5:コードの種類, 6〜17:コードトーン
        NoteInfo = copy.deepcopy(thisNote)
        if NoteInfo[0] != 128:  # midi番号（thisNote[0]）が音符のとき　
            NoteInfo[0] += Value
        # コードのルート音を転調
        NoteInfo[4] += Value
        NoteInfo[4] %= 12
        # コードトーンを転調
        NoteInfo[6:] = ChordSlide(NoteInfo[6:], Value)
        Output.append(NoteInfo)
    return Output

#　コード進行情報を　Value分　転調
def ChordModulation(ChordData, Value):
    Output = []  # 転調した音符情報の保存先
    if Value == 0:
        return ChordData
    for thisChord in ChordData:
        #ChordInfo↓
        # 0:コードのルート, 1:コードの種類, 2,:音価 3:Offset, 4:小節数, 5〜16:コードトーン
        ChordInfo = copy.deepcopy(thisChord)
        # コードのルート音　を　転調
        ChordInfo[0] += Value
        ChordInfo[0] %= 12
        # コードトーン　を　転調
        ChordInfo[5:] = ChordSlide(ChordInfo[5:], Value)
        Output.append(ChordInfo)
    return Output

#　コードトーンの音名リスト を　Value分　転調（シフト）
# [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]　
# ↓ Value = 2
# [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]　
def ChordSlide(Chord, Value):
    Chord = np.array(Chord)
    Chord = np.roll(Chord, Value)
    return Chord.tolist()

#　キーを　Value分　転調
def KeyModulation(KEY, Value):
    Key2ClassDict = {'C':0, 'C#':1, 'D-':1, 
                    'D':2, 'D#':3, 'E-':3, 
                    'E':4, 'E#':5, 'F-':4, 
                    'F':5, 'F#':6, 'G-':6, 
                    'G':7, 'G#':8, 'A-':8, 
                    'A':9, 'A#':10, 'B-':10,
                    'B':11, 'B#':0, 'C-':11}

    Class2KeyDict = ['C', 'Db', 'D', 'Eb',
                    'E', 'F', 'Gb', 'G',
                    'Ab', 'A', 'Bb', 'B']

    KeyID = Key2ClassDict[KEY]+Value
    OutputKey = Class2KeyDict[KeyID%12]
    return OutputKey
