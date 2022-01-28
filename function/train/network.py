import math
import torch
import torch.nn as nn

class MusicModel(nn.Module):
    def __init__(self, hChordRoot, hChordKind, hPitch, hDur, hOff,
                 ninp, nhid, nlayers, dropout=0.5, device='cpu'):
        super(MusicModel, self).__init__()
        # ドロップアウト
        self.drop = nn.Dropout(dropout)
        # コード進行情報用のLSTM層
        self.encStrChord = ChordRnnModel(hChordRoot, hChordKind, 
                                         ninp, nhid, nlayers, dropout=dropout)
        # 旋律情報用のLSTM層
        self.encNote = NoteRnnModel(hChordRoot, hChordKind, hPitch, hDur, hOff, 
                                    ninp, nhid, nlayers, dropout=dropout)
        # Masked self-Attention層
        nhead = 4
        self.d_model = nhid*2
        encoder_layers = nn.TransformerEncoderLayer(self.d_model, nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # 出力用の線形変換層
        self.decPitch = nn.Linear(nhid*2, hPitch)
        self.decDur = nn.Linear(nhid*2, hDur)
        # 重みの初期化
        self.init_weights()
        # 各パラメータ
        self.hPitch = hPitch
        self.hDur = hDur
        self.device = device
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):  # 重みの初期化
        initrange = 0.1
        nn.init.zeros_(self.decPitch.weight)
        nn.init.zeros_(self.decDur.weight)
        nn.init.uniform_(self.decPitch.weight, -initrange, initrange)
        nn.init.uniform_(self.decDur.weight, -initrange, initrange)

    def forward(self, input, hidden, strchord):
        # 旋律情報
        output_note, hidden = self.encNote(input, hidden)
        # コード進行情報
        output_chord = self.encStrChord(strchord)
        # コード進行情報と　旋律情報の 結合
        output = torch.cat((output_chord, output_note), -1)
        output = self.drop(output)
        # Masked self-attention
        output = output * math.sqrt(self.d_model)
        output_mask = self.generate_square_subsequent_mask(output)
        output = self.transformer_encoder(output, output_mask)
        # 音名と　音価の　出力
        decPitch = self.decPitch(output)
        decDur = self.decDur(output)
        decPitch = decPitch.view(-1, self.hPitch)
        decDur = decDur.view(-1, self.hDur)
        return decPitch, decDur, hidden

    def init_hidden(self, bsz): # 初期の隠れ層の重み
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

    def generate_square_subsequent_mask(self, data): 
        # Masked self-Attentionなので　マスクを作成
        nlen = data.size(0)
        output = torch.triu(torch.ones(nlen, nlen) * float('-inf'), diagonal=1)
        return output.to(self.device)


class NoteRnnModel(nn.Module):
    def __init__(self, hChordRoot, hChordKind, hPitch, hDur, hOff,
                 ninp, nhid, nlayers, dropout=0.5):
        super(NoteRnnModel, self).__init__()
        # DropOut
        self.drop = nn.Dropout(dropout)
        # Embedding層
        self.encChordRoot = nn.Embedding(hChordRoot, ninp)
        self.encChordKind = nn.Embedding(hChordKind, ninp)
        self.encPitch = nn.Embedding(hPitch, ninp)
        self.encDur = nn.Embedding(hDur, ninp)
        self.encOff = nn.Embedding(hOff, ninp)
        # LSTM層
        self.rnn = self.rnn = getattr(nn, 'LSTM')(ninp*5, nhid, nlayers, dropout=dropout)
        # 重みの初期化
        self.init_weights()
        # 各種パラメータ
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encChordRoot.weight, -initrange, initrange)
        nn.init.uniform_(self.encChordKind.weight, -initrange, initrange)
        nn.init.uniform_(self.encPitch.weight, -initrange, initrange)
        nn.init.uniform_(self.encDur.weight, -initrange, initrange)
        nn.init.uniform_(self.encOff.weight, -initrange, initrange)

    def forward(self, input, hidden):
        # 旋律情報の　エンコード
        chordRoot = self.drop(self.encChordRoot(input[0]))
        chordKind = self.drop(self.encChordKind(input[1]))
        Pitch = self.drop(self.encPitch(input[2]))
        Duration = self.drop(self.encDur(input[3]))
        Offset = self.drop(self.encOff(input[4]))
        # 曲の旋律　情報の　結合
        emb = torch.cat((chordRoot, chordKind, Pitch, Duration, Offset), -1)
        # LSTM層に入力
        output_note, hidden = self.rnn(emb, hidden)
        return output_note, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


class ChordRnnModel(nn.Module):
    def __init__(self, hChordRoot, hChordKind, ninp, nhid, nlayers, dropout=0.5):
        super(ChordRnnModel, self).__init__()
        # DropOut
        self.drop = nn.Dropout(dropout)
        # Embedding層
        self.encStrChordRoot = nn.Embedding(hChordRoot, ninp)
        self.encStrChordKind = nn.Embedding(hChordKind, ninp)
        # LSTM層
        self.rnn = self.rnn = getattr(nn, 'LSTM')(ninp*2, nhid, nlayers, dropout=dropout)
        # 重みの初期化
        self.init_weights()
        # 各種パラメータ
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encStrChordRoot.weight, -initrange, initrange)
        nn.init.uniform_(self.encStrChordKind.weight, -initrange, initrange)

    def forward(self, StrChord):
        batchsize = StrChord.size(2)                  # バッチサイズ
        # 各音符ごとにコード進行を読み込むために変形
        StrChord = StrChord.permute(0, 3, 1, 2).view(2, 6, -1)
        hidden = self.init_hidden(StrChord.size(-1))  # 初期入力の隠れ層の重み
        # コード進行情報の　エンコード
        StrchordRoot = self.drop(self.encStrChordRoot(StrChord[0]))
        StrchordKind = self.drop(self.encStrChordKind(StrChord[1]))
        # コード進行情報の　結合
        emb = torch.cat((StrchordRoot, StrchordKind), -1)
        # LSTM層
        output, hidden = self.rnn(emb, hidden)
        # 変形したのを元に戻す
        output = output.view(6, -1, batchsize, self.nhid)[-1]
        return output

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
