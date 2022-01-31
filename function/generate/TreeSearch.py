from anytree import Node
import torch
import copy
import pandas as pd
import os

def repackage_hidden(h):  # 最初の特徴量をコピーして切り離す
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class TreeSearch():
    def __init__(self, model, set_chord, data2tensor, rule, datasave, measure, name, outpath, OutputExtension='xml', Tempo=160):
        self.model = model
        # self.rule = rule
        self.set_chord = set_chord
        self.data2tensor = data2tensor
        self.rule = rule
        self.datasave = datasave
        self.measure = measure
        self.name = name
        self.outpath = outpath
        self.OutputExtension = OutputExtension
        self.Tempo = Tempo
        os.makedirs('{}/csv/'.format(outpath), exist_ok=True)
        self.flag = False

    def generate(self, Pitch, Duration, Offset):
        PitchList = [Pitch]
        DurationList = [Duration]
        OffsetList = [Offset]
        DataList, DataStrList = [], []
        note_num = 1
        Chord, ChordStr = self.set_chord.OutChord(Offset)[0], self.set_chord.OutChord(Offset)[1]
        Data, DataStr = self.data2tensor.converter(ChordStr, Chord, Pitch, Duration, Offset)
        DataList.append(Data), DataStrList.append(DataStr)
        root = Node('note{}_(input:{}_{}_{})'.format(note_num, Pitch, int(Duration), int(Offset % 1920)),
                    parent=None, Pitch=PitchList, Duration=DurationList, Offset=OffsetList, 
                    PitchIdx=['Input'], DurationIdx=['Input'],
                    Note = 'P:{}_D:{}_O:{}_M:{}'.format(Pitch, int(Duration), int(Offset % 1920), int(Offset/1920)+1))
        self.generate_node(root, DataList, DataStrList, note_num,
                           PitchList, DurationList, OffsetList, ['Input'], ['Input'])
        return root

    def generate_node(self, root, DataList, DataStrList, note_num, PitchList, DurationList, OffsetList, SavePitchIdxList, SaveDurationIdxList):
        if self.flag:
            return
        PitchListNode, DurationListNode, OffsetListNode = copy.deepcopy(PitchList), copy.deepcopy(DurationList), copy.deepcopy(OffsetList)
        DataListNode, DataStrListNode = copy.deepcopy(DataList), copy.deepcopy(DataStrList)
        SavePitchIdxListNode, SaveDurationIdxListNode = copy.deepcopy(SavePitchIdxList), copy.deepcopy(SaveDurationIdxList)
        Duration, Offset = DurationListNode[-1], OffsetListNode[-1]
        hidden = self.model.init_hidden(1)
        hidden = repackage_hidden(hidden)
        Data, DataStr = self.SetDATA(DataList, DataStrList)
        OutPitch, OutDuration, _ = self.model(Data, hidden, DataStr)
        NextPitchList, NextDuration, NextOffset, PitchIdxList, DurationIdx = self.rule.Select(OutPitch, OutDuration, Offset, Duration)
        if len(NextPitchList) == 0:return
        DurationListNode.append(NextDuration)
        OffsetListNode.append(NextOffset)
        note_num += 1
        children = []
        for PitchIdx, NextPitch in zip(PitchIdxList, NextPitchList):
            children.append(Node('note{}_(P_idx:{},D_idx:{})'.format(note_num, PitchIdx, DurationIdx), 
                                 parent=root, Pitch=PitchListNode+[NextPitch], Duration=DurationListNode, Offset=OffsetListNode, 
                                 PitchIdx=SavePitchIdxListNode+[PitchIdx], DurationIdx=SaveDurationIdxListNode+[DurationIdx],
                                 Note='P:{}_D:{}_O:{}_M:{}'.format(NextPitch, int(NextDuration), int(NextOffset%1920), int(NextOffset/1920)+1)))
            if NextOffset + NextDuration >= self.measure*1920:
                self.flag = True
                df = pd.DataFrame([['PitchIdx:']+SavePitchIdxListNode+[PitchIdx], 
                                   ['DurationIdx:']+SaveDurationIdxListNode+[DurationIdx]])
                df.to_csv('{}/csv/{}.csv'.format(self.outpath, self.name))
                self.datasave.SetNote(PitchListNode+[NextPitch], DurationListNode)
                self.datasave.Output(self.name, self.outpath, Type=self.OutputExtension, Tempo=self.Tempo, Musescore=False)
                break
        for idx, child in enumerate(children):
            NextPitch = NextPitchList[idx]
            NextPitchIdx = PitchIdxList[idx]
            Chord, ChordStr= self.set_chord.OutChord(NextOffset)[0], self.set_chord.OutChord(NextOffset)[1]
            Data, DataStr = self.data2tensor.converter(ChordStr, Chord, NextPitch, NextDuration, NextOffset)
            self.generate_node(child, DataListNode+[Data], DataStrListNode+[DataStr], note_num,
                               PitchListNode+[NextPitch], DurationListNode, OffsetListNode, 
                               SavePitchIdxListNode+[NextPitchIdx], SaveDurationIdxListNode+[DurationIdx])
    
    def SetDATA(self, DataList, DataStrList):
        DataList = copy.deepcopy(DataList)
        DataStrList = copy.deepcopy(DataStrList)
        Data = torch.cat(DataList, dim=1)
        DataStr = torch.cat(DataStrList, dim=1)
        return Data, DataStr
