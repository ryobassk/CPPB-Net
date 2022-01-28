from function.make_dataset import *
import glob
import os
import numpy as np
import shutil
import argparse

#  パラメータ
parser = argparse.ArgumentParser(description='BebopNetのDatasetを作成')
parser.add_argument('--data', type=str, default='data/charlie/',
                    help='xmlデータのパス')
parser.add_argument('--output', type=str, default='dataset/charlie/',
                    help='出力先のパス')
parser.add_argument("--modulation_Flag", action='store_false',
                    help='Trueで転調させる')
parser.add_argument("--modulation_BeginIdx", type=int, default=-12,
                    help='転調の範囲（開始位置）')
parser.add_argument("--modulation_EndIdx", type=int, default=13,
                    help='転調の範囲（終了位置）')
args = parser.parse_args()

#  データセットの出力先
if os.path.exists(args.output):
    shutil.rmtree(args.output)
os.makedirs(args.output, exist_ok=True)
OutPath = {'note': '{}/note/'.format(args.output),   # 音符情報の出力先
           'chord': '{}/chord/'.format(args.output)   # コード進行情報の出力先
           }
os.makedirs(OutPath['note'], exist_ok=True), os.makedirs(OutPath['chord'], exist_ok=True)

#　転調の範囲を設定
if args.modulation_Flag:
    # 転調時
    Modulation_Range = np.arange(args.modulation_BeginIdx,
                                 args.modulation_EndIdx)
else:
    Modulation_Range = np.array([0])

#　xmlファイルのパスを取得
FilePath_List = glob.glob('{}/*.xml'.format(args.data))

#　データセットの作成
for FilePath in FilePath_List:
    # パス(FilePath)のデータ名
    FileName = os.path.splitext(os.path.basename(FilePath))[0]
    # パスのデータ取得（音符情報、コード進行、キー）
    Notes, Chords, Key = XmlLoad(FilePath)
    for value in Modulation_Range:
        # 音符の転調
        ModulationNotes = NoteModulation(Notes, value)
        # コードの音程リストの転調
        ModulationChords = ChordModulation(Chords, value)
        # 転調した際のキー
        ModulationKey = KeyModulation(Key, value)
        # データをcsvとして保存
        SaveData(ModulationNotes, '{}/{}_{}~{}.csv'.format(
            OutPath['note'], ModulationKey, value, FileName), Type='Note')
        SaveData(ModulationChords, '{}/{}_{}~{}.csv'.format(
            OutPath['chord'], ModulationKey, value, FileName), Type='Chord')
