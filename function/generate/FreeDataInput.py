def YesOrNoInput():
    inputdata = str(input('y or n を入力してください：'))
    if inputdata == 'y' or inputdata == 'n':
        return inputdata
    else:
        print('入力が間違っています。もう一度入力してください。')
        YesOrNoInput()

def ChordInput(Measure, BeatCount, UsedChordRootList, UsedChordKindList):
    ChordRoot = str(input('{}小節目{}拍目のコード(ルート)：'.format(Measure+1, BeatCount)))
    if not ChordRoot in UsedChordRootList:
        print('入力したコード(ルート)は使用できません')
        print('以下のコード(ルート)が使用可能です\n', UsedChordRootList, '\n')
        return ChordInput(Measure, BeatCount, UsedChordRootList, UsedChordKindList)
    ChordKind = str(input('{}小節目{}拍目のコード(シンボル)：'.format(Measure+1, BeatCount)))
    if not ChordKind in UsedChordKindList:
        print('入力したコード(シンボル)は使用できません')
        print('以下のコード(シンボル)が使用可能です\n', UsedChordKindList, '\n')
        return ChordInput(Measure, BeatCount, UsedChordRootList, UsedChordKindList)
    print()
    return ChordRoot, ChordKind


def ChordProgressionInput(UsedChordRootList, UsedChordKindList):
    ChordProgression_root = []
    ChordProgression_kind = []
    Chord_Num = int(input('楽曲の小節数：'))
    print()
    print('使用可能なコード')
    print('・ルート', UsedChordRootList)
    print()
    print('・シンボル', UsedChordKindList)
    print()
    for Measure in range(Chord_Num):
        ChordRoot_Measure1, ChordKind_Measure1 = ChordInput(Measure, 1, UsedChordRootList, UsedChordKindList)
        ChordRoot_Measure3, ChordKind_Measure3 = ChordInput(Measure, 3, UsedChordRootList, UsedChordKindList)
        ChordProgression_root += [ChordRoot_Measure1, ChordRoot_Measure3]
        ChordProgression_kind += [ChordKind_Measure1, ChordKind_Measure3]
    ChordProgression = []
    for root, kind in zip(ChordProgression_root, ChordProgression_kind):
        ChordProgression.append('{}_{}'.format(root, kind))
    print('下記のコード進行でよろしいですか？\n', ChordProgression)
    cheakflag = YesOrNoInput()
    if cheakflag == 'y':
        return ChordProgression_root, ChordProgression_kind
    else:
        ChordProgressionInput(UsedChordRootList, UsedChordKindList)

def PitchInput(NoteIdx, UsedPitchList):
    Pitch = str(input('{}つ目の音符（音程）：'.format(NoteIdx+1)))
    if Pitch in UsedPitchList:
        return Pitch
    else:
        print('入力した音程は使用できません')
        print('以下の音程が使用可能です\n', UsedPitchList, '\n')
        return PitchInput(NoteIdx, UsedPitchList)

def DurationInput(NoteIdx, UsedDurationNumList):
    Duration = int(input('{}つ目の音符（音価）：'.format(NoteIdx+1)))
    if Duration in UsedDurationNumList:
        return Duration
    else:
        print('入力した音価は使用できません')
        print('以下の音価が使用可能です\n', UsedDurationNumList, '\n')
        DurationInput(NoteIdx, UsedDurationNumList)

def NoteInput(NoteIdx, UsedPitchList, UsedDurationNumList):
    Pitch = PitchInput(NoteIdx, UsedPitchList)
    Duration = DurationInput(NoteIdx, UsedDurationNumList)
    print()
    return '{}-{}'.format(Pitch, Duration), Pitch, Duration


def NotesInput(UsedPitchList, UsedDurationNumList, Note_Num=None):
    Notes = []
    PitchList, DurationList = [], []
    if Note_Num==None:
        Note_Num = int(input('入力する音符数：'))
        print()
    print('使用可能な音符')
    print('・音程（音名_オクターブ、国際式、休符はRest）\n', UsedPitchList)
    print()
    print('・音価、（例）480=4分音符、240=8分音符\n',  UsedDurationNumList)
    print()
    for NoteIdx in range(Note_Num):
        InputNoteData, Pitch, Duration = NoteInput(NoteIdx, UsedPitchList, UsedDurationNumList)
        Notes += [InputNoteData]
        PitchList+=[Pitch]
        DurationList+=[Duration]
    print('下記の入力音符でよろしいですか？\n', Notes)
    cheakflag = YesOrNoInput()
    if cheakflag == 'y':
        return PitchList, DurationList
    else:
        if Note_Num == None:
            NotesInput(UsedPitchList, UsedDurationNumList)
        else:
            NotesInput(UsedPitchList, UsedDurationNumList, Note_Num=1)

def Chord2Data(roots, kinds):
    SetRoot = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3,
               'E': 4, 'F': 5, 'Gb': 6, 'G': 7,
               'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}

    SetKind = {'dim':'diminished', '7':'dominant-seventh',
               'm7b5':'half-diminished-seventh', 'M':'major',
               '6':'major-sixth', 'm':'minor'}
    out_roots, out_kinds = [], []
    for root, kind in zip (roots, kinds):
        out_roots.append(SetRoot[root])
        out_kinds.append(SetKind[kind])
    return out_roots, out_kinds
def Pitch2Data(Pitchs):
    SetRoot = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3,
               'E': 4, 'F': 5, 'Gb': 6, 'G': 7,
               'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
    out_pitchs = []
    for pitch in Pitchs:
        if pitch == 'Rest':
            out_pitchs.append(128)
        else:
            root, octave = pitch.split('_')
            out_pitchs.append(SetRoot[root]+(int(octave)+1)*12)
    return out_pitchs

class InputFreeData(object):
    def __init__(self,):
        self.UsedPitchList = ['D_2', 'Eb_2', 'E_2', 'F_2', 'Gb_2', 'G_2', 'Ab_2', 'A_2',
                              'Bb_2', 'B_2', 'C_3', 'Db_3', 'D_3', 'Eb_3', 'E_3', 'F_3',
                              'Gb_3', 'G_3', 'Ab_3', 'A_3', 'Bb_3', 'B_3', 'C_4', 'Db_4',
                              'D_4', 'Eb_4', 'E_4', 'F_4', 'Gb_4', 'G_4', 'Ab_4', 'A_4',
                              'Bb_4', 'B_4', 'C_5', 'Db_5', 'D_5', 'Eb_5', 'E_5', 'F_5',
                              'Gb_5', 'G_5', 'Ab_5', 'A_5', 'Bb_5', 'B_5', 'C_6', 'Db_6',
                              'D_6', 'Eb_6', 'E_6', 'F_6', 'Gb_6', 'G_6', 'Ab_6', 'A_6', 
                              'Ab_9', 'Rest']
        # UsedPitchList = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        #                  50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        #                  62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
        #                  74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
        #                  86, 87, 88, 89, 90, 91, 92, 93, 128]
        self.UsedDurationNumList = [40, 60, 80, 96, 120, 160,
                                    240, 320, 360, 384, 480,
                                    720, 960, 1440, 1920]
        self.UsedChordRootList = ['C', 'Db', 'D', 'Eb',
                                  'E', 'F', 'Gb', 'G',
                                  'Ab', 'A', 'Bb', 'B']
        # self.UsedChordKindList = ['M', 'm', '7', 'dim', 'm7b5', '6']
        self.UsedChordKindList = ['M', 'm', '7', 'dim', 'm7b5']
    def SetData(self,ModelName, PathName):
        Chords_root, Chords_kind = ChordProgressionInput(self.UsedChordRootList, self.UsedChordKindList)
        Pitchs, Durations = NotesInput(self.UsedPitchList, self.UsedDurationNumList, Note_Num=1)
        Chords_root, Chords_kind = Chord2Data(Chords_root, Chords_kind)
        Pitchs = Pitch2Data(Pitchs)
        return [[[Chords_root, Chords_kind], Pitchs, Durations]], ['FreeGenerateMusic_{}_{}'.format(ModelName, PathName)]
