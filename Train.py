# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import DataLoader as DL

from function.train.DataSet import TrainDataset, padding_seq
from function.train.DataObject import Corpus
import function.train.network as net
import function.train.train_function as fn

from tqdm import tqdm
import argparse
import time
import math
import os
import sys
import pickle


#パラメータ
parser = argparse.ArgumentParser(description='SSMGの訓練プログラム')
parser.add_argument("--debag_Flag", action='store_true',
                    help='デバッグするか')
parser.add_argument('--data', type=str, default='dataset/charlie',
                    help='location of the data')
parser.add_argument('--Outpath', type=str, default='.',
                    help='OUTPUT path')
parser.add_argument('--LogName', type=str, default='charlie',
                    help='OUTPUT FOLDER NAME')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=300,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str, default='cuda',
                    help='use CUDA')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
args = parser.parse_args()

#  テスト時のデータ設定
if args.debag_Flag:
    print('TEST MODE')
    args.data, args.LogName = 'dataset/Test', 'Test'

# ランダムシードの設定
torch.manual_seed(args.seed)

# CPU or GPUの設定
if torch.cuda.is_available():
    print('Use GPU')
    device = torch.device("cuda" if args.cuda else "cpu")
else:
    print('Use CPU')
    device = torch.device("cpu")

#　出力先のフォルダ作成
OutFolder = '{}/log/{}/'.format(args.Outpath, args.LogName)
os.makedirs(OutFolder, exist_ok=True)
os.makedirs(OutFolder+'model_epoch', exist_ok=True)

###############################################################################
#データの読み込み
###############################################################################
corpus = Corpus(args.data)
train_dataset = TrainDataset(corpus.train)
test_dataset = TrainDataset(corpus.test)

with open(OutFolder+'Dictionary.pickle', mode='wb') as f:
    pickle.dump(corpus.Dict, f)
with open(OutFolder+'args.pickle', mode='wb') as f:
    pickle.dump(args, f)

dataloader = {'train': DL(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=os.cpu_count(),
                          pin_memory=True,
                          collate_fn=padding_seq),
              'test': DL(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=os.cpu_count(),
                         pin_memory=True,
                         collate_fn=padding_seq
                         )}

###############################################################################
#モデルを構築
###############################################################################
hChordRoot = len(corpus.Dict.DictChordRoot)
hChordKind = len(corpus.Dict.DictChordKind)
hPitch = len(corpus.Dict.DictPitch)
hDuration = len(corpus.Dict.DictDuration)
hOffset = len(corpus.Dict.DictOffset)
#モデルの設定
model = net.MusicModel(hChordRoot=hChordRoot,
                       hChordKind=hChordKind,
                       hPitch=hPitch,
                       hDur=hDuration,
                       hOff=hOffset,
                       ninp=args.emsize,
                       nhid=args.nhid,
                       nlayers=args.nlayers,
                       dropout=args.dropout,
                       device=device).to(device)
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
optimizer = optimizers.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# 途中から学習する際の学習済みモデルの読み込み
#path_model = './log/6_15/model_epoch2.pt'
#model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))

###############################################################################
#訓練プログラム
###############################################################################
def train():
    start_time = time.time()
    model.train()
    TotalLoss, PitchLoss, DurationLoss = 0., 0., 0.
    with tqdm(total=len(dataloader['train']), unit="batch") as pbar:
        pbar.set_description(f"Epoch[{epoch}/{args.epochs}]({'train'})")
        epoch_idx = 1
        for batchIdx, batchData in enumerate(dataloader['train']):
            data, targets, strchord = fn.get_batch(batchData, device)
            hidden = model.init_hidden(data.size(-1))
            hidden = fn.repackage_hidden(hidden)
            Predict_pitch, Predict_duration, _ = model(data, hidden, strchord)

            lossPitch = criterion(Predict_pitch, targets[0])
            lossDuration = criterion(Predict_duration, targets[1])
            loss = lossPitch + lossDuration

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            TotalLoss += loss.item()
            PitchLoss += lossPitch.item()
            DurationLoss += lossDuration.item()
            aveloss = TotalLoss / epoch_idx
            aveloss_pitch, aveloss_duration = PitchLoss / epoch_idx, DurationLoss / epoch_idx
            epoch_idx += 1

            pbar.set_postfix({'loss:': aveloss,
                              'pitch:': aveloss_pitch,
                              'Duration:': aveloss_duration,
                              'ppl:': math.exp(aveloss),
                              'time:': time.time() - start_time,
                              'lr:': scheduler.get_last_lr()[0]})
            pbar.update(1)

            if torch.isnan(loss) == True:sys.exit() # lossが発散した時は終了
        return aveloss, aveloss_pitch, aveloss_duration


def evaluate():  # テストデータでの評価
    start_time = time.time()
    model.eval()
    TotalLoss, PitchLoss, DurationLoss = 0., 0., 0.
    with tqdm(total=len(dataloader['test']), unit="batch") as pbar:
        pbar.set_description(f"Epoch[{epoch}/{args.epochs}]({'test'})")
        with torch.no_grad():
            epoch_idx = 1
            for batchIdx, batchData in enumerate(dataloader['test']):
                data, targets, strchord = fn.get_batch(batchData, device)
                
                hidden = model.init_hidden(data.size(-1))
                hidden = fn.repackage_hidden(hidden)
                Predict_pitch, Predict_duration, _ = model(data, hidden, strchord)

                lossPitch = criterion(Predict_pitch, targets[0])
                lossDuration = criterion(Predict_duration, targets[1])
                loss = lossPitch + lossDuration
                TotalLoss += loss.item()
                PitchLoss += lossPitch.item()
                DurationLoss += lossDuration.item()
                aveloss = TotalLoss / epoch_idx
                aveloss_pitch, aveloss_duration = PitchLoss / epoch_idx, DurationLoss / epoch_idx
                epoch_idx += 1

                pbar.set_postfix({'loss:': aveloss,
                                'pitch:': aveloss_pitch,
                                'Duration:': aveloss_duration,
                                'ppl:': math.exp(aveloss),
                                'time:': time.time() - start_time,
                                'lr:': scheduler.get_last_lr()[0]})
                pbar.update(1)
            return aveloss, aveloss_pitch, aveloss_duration


#訓練と評価のループ
# 損失の保存場所を作成
loss_value = {"train": [], "val": []}
pitchloss_value = {"train": [], "val": []}
durationloss_value = {"train": [], "val": []}
best_value = {"epoch": 1, "loss": None}
Names = ['loss', 'loss_pitch', 'loss_duration']
try:
    for epoch in range(1, args.epochs+1):
        train_loss = train()
        val_loss = evaluate()
        # loss
        loss_value['train'].append(train_loss[0])
        loss_value['val'].append(val_loss[0])
        # pitch loss
        pitchloss_value['train'].append(train_loss[1])
        pitchloss_value['val'].append(val_loss[1])
        # duration loss
        durationloss_value['train'].append(train_loss[2])
        durationloss_value['val'].append(val_loss[2])
        # 最もval_lossが小さいモデルを保存
        if not best_value['loss'] or val_loss[0] < best_value['loss']:
            torch.save(model.state_dict(), '{}/model.pt'.format(OutFolder))
            best_value['epoch'], best_value['loss'] = epoch, val_loss[0]
        else:
            scheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), '{}/model_epoch/model_epoch{}.pt'.format(OutFolder, epoch))
        fn.PlotDef(loss_value,
                   pitchloss_value,
                   durationloss_value,
                   Names,
                   best_value,
                   OutFolder)
except KeyboardInterrupt:
    print('-' * 89)
    print('ERROR or TrainingStop')
