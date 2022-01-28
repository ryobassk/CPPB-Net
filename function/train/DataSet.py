from torch.utils import data
import torch

class TrainDataset(data.Dataset):
    def __init__(self, input_data):
        self.input_data = input_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        #  コード進行　情報 --> self.input_data[index][0]
        StrchordRoot = torch.tensor(self.input_data[index][0]).permute(1, 0, 2)[0]
        StrchordKind = torch.tensor(self.input_data[index][0]).permute(1, 0, 2)[1]

        #  曲の旋律　情報 --> self.input_data[index][1]
        chordRoot = torch.tensor(self.input_data[index][1][0])
        chordKind = torch.tensor(self.input_data[index][1][1])
        note = torch.tensor(self.input_data[index][1][2])
        duration = torch.tensor(self.input_data[index][1][3])
        offset = torch.tensor(self.input_data[index][1][4])
        return StrchordRoot, StrchordKind,  chordRoot, chordKind, note, duration, offset

def padding_seq(batch):#バッチ処理（長さを０で揃える）
    #  コード進行　情報 --> item[0]〜item[3] --> Chord_info
    StrchordRoot = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True)
    StrchordKind = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True)
    Chord_info = torch.stack([StrchordRoot , StrchordKind], dim=0)

    #  曲の旋律　情報 --> item[4]〜item[10] --> Note_info
    chordRoot = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True)
    chordKind = torch.nn.utils.rnn.pad_sequence([item[3] for item in batch], batch_first=True)
    note = torch.nn.utils.rnn.pad_sequence([item[4] for item in batch], batch_first=True)
    duration = torch.nn.utils.rnn.pad_sequence([item[5] for item in batch], batch_first=True)
    offset = torch.nn.utils.rnn.pad_sequence([item[6] for item in batch], batch_first=True)
    Note_info = torch.stack([chordRoot, chordKind, note, duration, offset], dim=0)
    return [Chord_info, Note_info]
