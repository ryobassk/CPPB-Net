import torch
import matplotlib.pyplot as plt
import csv

def repackage_hidden(h):  # 最初の特徴量をコピーして切り離す
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(batchData, device):  # バッチデータをStrChord, input, targetの3つに分ける
    input = batchData[1][:, :, :-1].permute(0, 2, 1).to(device)
    target = batchData[1][2:4, :, 1:].permute(0, 2, 1).to(device).view(2, -1)
    StrChord = batchData[0][:, :, :-1, :].permute(0, 2, 1, 3).to(device)
    return input, target, StrChord

#　結果の出力
# lossの保存とグラフの作成
def PlotDef(AllLoss, PitchLoss, DurationLoss, Names, BestValue, Folder):  
    TrainLoss = [AllLoss['train'], PitchLoss['train'], DurationLoss['train']]
    ValLoss = [AllLoss['val'], PitchLoss['val'], DurationLoss['val']]
    with open('{}/Loss.csv'.format(Folder), "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['BestEpoch:', BestValue['epoch'], 'Bestloss:', BestValue['loss']])
        for TrainData, ValData, Name in zip(TrainLoss, ValLoss, Names):
            plt.plot(range(1, len(TrainData) + 1), TrainData, label='Train')
            plt.plot(range(1, len(ValData) + 1), ValData, label='Test')
            plt.ylabel("CROSS_ENTROPY_ERROR")
            plt.xlabel("EPOCH")
            plt.title("Loss")
            plt.legend()
            plt.savefig('{}/{}loss.png'.format(Folder, Name))
            plt.clf()
            writer.writerow([x for x in range(len(TrainData)+1)])
            writer.writerow(['{}/Train'.format(Name)]+TrainData)
            writer.writerow(['{}/Test'.format(Name)]+ValData)
            writer.writerow([])

