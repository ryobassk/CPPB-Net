# ライブラリ
import music21 as m21
import pretty_midi
import torch

#　データの変換
class Data2Tensor(object):
    def __init__(self, Corpus, Device):
        self.corpus = Corpus
        self.device = Device

    def converter(self, STRCHORD, CHORD,  PITCH, DURATION, TICK):
        OFFSET = TICK % 1920
        ChordRoot, ChordKind = self.corpus.DictChordRoot.word2idx[CHORD[0]
                                                                  ], self.corpus.DictChordKind.word2idx[CHORD[1]]
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
                               'major': '', 'half-diminished-seventh': 'm7b5',
                               'major-sixth': '6', 'diminished':'dim'}

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
                NoteSymbol = m21.note.Note(
                    pitch, quarterLength=quarterLength, offset=OffsetLength)
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
        elif Type == 'both':  # MIDｒｔIファイルで保存
            GenerateData.write('musicxml', '{}/{}.xml'.format(Path, Name))
            GenerateData.write('midi', '{}/{}.mid'.format(Path, Name))

    def XML2MIDI(self, DATAXML, Tempo):  # xmlからコード情報を除くMIDIを作成
        generateData1 = m21.stream.Stream()  # 楽譜オブジェクトの生成
        generateData1.append(m21.clef.TrebleClef())  # ト音記号に設定
        generateData1.append(m21.meter.TimeSignature('4/4'))  # 四分の四に設定
        generateData2 = m21.stream.Stream()  # 楽譜オブジェクトの生成
        generateData2.append(m21.clef.TrebleClef())  # ト音記号に設定
        generateData2.append(m21.meter.TimeSignature('4/4'))  # 四分の四に設定
        musicdata = m21.converter.parse(DATAXML)
        for thisNote in musicdata.flat.notesAndRests:  # musicdata.flat.notes:
            if thisNote.isNote:
                note = m21.note.Note(
                    thisNote.pitch.midi, quarterLength=thisNote.duration.quarterLength, offset=thisNote.offset, velocity=100)
                generateData1.append(note)
            elif thisNote.isRest:
                note = m21.note.Rest(
                    quarterLength=thisNote.duration.quarterLength, offset=thisNote.offset)
                generateData1.append(note)
            elif thisNote.isChord:
                generateData2.append(thisNote)
        generateData1.makeMeasures(inPlace=True)  # 小節追加
        generateData1.write('midi', 'Test1.mid')
        generateData2.makeMeasures(inPlace=True)  # 小節追加
        generateData2.write('midi', 'Test2.mid')
        self.Midiout('Test1.mid', 'Test2.mid', Tempo, 'Test3.mid')
    
    def Midiout(self, Datamelo, Datachord, Tempo, OutMidi):
        #midiの作成、４分音符を480tick、BPM=160と設定
        musicdata = pretty_midi.PrettyMIDI(resolution=480,
                                            initial_tempo=Tempo)
        midi_data1 = pretty_midi.PrettyMIDI(Datamelo).instruments
        midi_data2 = pretty_midi.PrettyMIDI(Datamelo).instruments
        print(midi_data1)

        #各楽器の情報をMIDIに格納
        musicdata.instruments.append(midi_data1)
        musicdata.instruments.append(midi_data2)
        musicdata.write('cello-C-chord.mid')
        a=1
