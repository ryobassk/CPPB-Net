# coding: utf-8
#　ライブラリ
import copy
import torch
import argparse
import pickle
import os

import function.train.network as net
import function.generate.TreeSearch as Tree
import function.generate.RuleFunction as Rulefn
import function.generate.SetChord as SetChord
import function.generate.SetMusic as SetMusic
import function.generate.Dataset2Rule as Dataset2Rule

parser = argparse.ArgumentParser(description='CPPB')
parser.add_argument('--StartChordNum', type=int, default=1,
                    help='現在の小節から何小節前までのコードを使用するか')
parser.add_argument('--EndChordNum', type=int, default=1,
                    help='現在の小節から何小節後までのコードを使用するか')
parser.add_argument('--StartUseMidiNum', type=int, default=0,
                    help='使用する音名の範囲(最低音)')
parser.add_argument('--EndUseMidiNum', type=int, default=127,
                    help='使用する音名の範囲(最高音)')
parser.add_argument('--OutputExtension', type=str, default='xml',
                    help='出力する拡張子')
parser.add_argument('--Tempo', type=int, default=160,
                    help='生成する楽曲のBPM')
parser.add_argument('--ChordProgressNum', type=int, default=1,
                    help='特定のコード進行を何回繰り返すか')
parser.add_argument('--PathName', type=str, default='sample',
                    help='読み込む学習済みモデルのフォルダ名')
parser.add_argument('--ModelName', type=str, default='CPPB',
                    help='生成結果の名前（ファイル名）')
parser.add_argument('--Logpath', type=str, default='log/',
                    help='読み込む学習済みモデルのフォルダのパス')
parser.add_argument('--Outpath', type=str, default='data_generate/',
                    help='生成結果の出力先フォルダのパス')
parser.add_argument('--RuleName', type=str, default='model_searchTree',
                    help='生成するルール(model_only or model_searchTree)')
parser.add_argument("--Dataset2Rule_Flag", action='store_true',
                    help='データセットからルールを作成するか')
parser.add_argument('--DatasetPath', type=str, default='dataset/charlie',
                    help='ルール作成に必要なデータセットのパス')
parser.add_argument('--DataSelect', type=str, default='DataSelect',
                    help='DataSelect:確率順に選択、生成、RandomDataSelect:確率に基づきランダムに選択、生成')
parser.add_argument('--Key', type=str, default='C',
                    help='生成するキー')
parser.add_argument('--ChordProgression', type=str, default='Two_Five_One',
                    help='生成するキー')
generate_args = parser.parse_args()

#パラメータ
StartChordNum = generate_args.StartChordNum
EndChordNum = generate_args.EndChordNum
UseMidiNum = [generate_args.StartUseMidiNum, generate_args.EndUseMidiNum]
ChordProgress_Num = generate_args.ChordProgressNum
RuleName = generate_args.RuleName
PathName = generate_args.PathName                          # 読み込む学習済みモデルのフォルダ名
ModelName = generate_args.ModelName                        # 生成結果の名前（ファイル名）
Logpath = generate_args.Logpath +'/{}/'.format(PathName)   # 読み込む学習済みモデルのフォルダのパス
Outpath = generate_args.Outpath + '/{}/{}/'.format(
                                PathName, RuleName)        # 生成結果の出力先フォルダのパス
os.makedirs(Outpath, exist_ok=True)
if generate_args.Key== 'All':
    KeyList = ['C', 'Db', 'D', 'Eb',
            'E', 'F', 'Gb', 'G',
            'Ab', 'A', 'Bb', 'B']                   # 生成するキー

else:
    KeyList = [generate_args.Key]                 # 生成するキー

if generate_args.ChordProgression == 'All':
    ChordProgressList = ['Two_Five_One',
                        'Autumn_Leaves',
                        'Just_The_Two_Of_Us',
                        'Blues1',
                        'Blues2']                     # 生成する音楽のコード進行
else:
    ChordProgressList = [generate_args.ChordProgression]

# 学習時のパラメータを取得
path_model = '{}/model.pt'.format(Logpath)
parser = argparse.ArgumentParser(description='CPPB')
with open('{}/args.pickle'.format(Logpath), mode='rb') as f:
    args = pickle.load(f)
with open('{}/Dictionary.pickle'.format(Logpath), mode='rb') as f:
    corpus = pickle.load(f)

# CPU or GPUの設定
if torch.cuda.is_available():
    print('Use GPU')
    device = torch.device("cuda" if args.cuda else "cpu")
else:
    print('Use CPU')
    device = torch.device("cpu")

#Classと文字の変換関数
SetKey = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3,
          'E': 4, 'F': 5, 'Gb': 6, 'G': 7,
          'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
SetKeyList = ['C', 'Db', 'D', 'Eb',
              'E', 'F', 'Gb', 'G',
              'Ab', 'A', 'Bb', 'B']
#ルールと関数の変換関数
RuleFunctionNameDict = {'model_only': 'PitchRule1',
                        'model_searchTree': 'PitchRule4',
                        }

#　入力設定
InputList = []                      # 生成する音楽の入力リスト
NameList = []                       # 生成する音楽の名前リスト
ChordProgressData = SetChord.ChordProgressData()
for ChordProgressFigure in ChordProgressList:
    for key in KeyList:
        if ChordProgressFigure=='Autumn_Leaves' or ChordProgressFigure=='Two_Five_One':
            indexkey = SetKey[key]
            indexkey = (indexkey + 2)%12
            inputnoteclass = SetKeyList[indexkey]
        else:
            inputnoteclass = key
        ChordProgress = ChordProgressData.SetChord(key, ChordProgressFigure, Num=ChordProgress_Num)
        InputList.append(SetChord.SetData(inputnoteclass, '8', ChordProgress))
        InputList.append(SetChord.SetData(inputnoteclass, '4', ChordProgress))
        InputList.append(SetChord.SetData('Rest', '8', ChordProgress))
        InputList.append(SetChord.SetData('Rest', '4', ChordProgress))
        NameList.append('{}_{}_Key-{}_Ver-{}_Note-{}_Len-{}'.format(ModelName, ChordProgressFigure, key, PathName, inputnoteclass, '8'))
        NameList.append('{}_{}_Key-{}_Ver-{}_Note-{}_Len-{}'.format(ModelName, ChordProgressFigure, key, PathName, inputnoteclass, '4'))
        NameList.append('{}_{}_Key-{}_Ver-{}_Note-{}_Len-{}'.format(ModelName, ChordProgressFigure, key, PathName, 'Rest', '8'))
        NameList.append('{}_{}_Key-{}_Ver-{}_Note-{}_Len-{}'.format(ModelName, ChordProgressFigure, key, PathName, 'Rest', '4'))


###############################################################################
#　モデルを構築
###############################################################################
hChordRoot = len(corpus.DictChordRoot)
hChordKind = len(corpus.DictChordKind)
hPitch = len(corpus.DictPitch)
hDuration = len(corpus.DictDuration)
hOffset = len(corpus.DictOffset)
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
model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))
model.eval()

###############################################################################
#　音楽生成
###############################################################################
# データセットからルールを作成
if generate_args.Dataset2Rule_Flag:
    dataset = Dataset2Rule.ExpandData(Dataset2Rule.CsvLoadOutput(
                '{}/note'.format(generate_args.DatasetPath), Artist=None))
    Dataset2RuleDict = Dataset2Rule.DataDict()
    Dataset2RuleDict.loaddata(dataset)
    f = open('{}/Dataset2RuleDict.pkl'.format(Logpath), 'wb')
    pickle.dump(Dataset2RuleDict.DictNote, f)
Dataset2RuleDict_Path = '{}/Dataset2RuleDict.pkl'.format(Logpath)

RuleFunctionName = RuleFunctionNameDict[RuleName]

#関数設定
#　データをtensorに変形する関数
Data2Tensor = SetMusic.Data2Tensor(corpus, device)
#　候補の中から音符を選択する関数
# 　'PitchRule':確率の中で一番高いPITCHを出力
DataSelect = getattr(Rulefn,
                     generate_args.DataSelect)(corpus, device,
                                               rulename=RuleFunctionName,            # 生成するときのルール
                                               startmidinum=UseMidiNum[0],   # 使用する音名の範囲(最低音)
                                               endmidinum=UseMidiNum[1],     # 使用する音名の範囲(最高音)
                                               Dataset2RuleDict_Path=Dataset2RuleDict_Path    # ルール辞書のパス
                                               )

print('音楽生成')
for musicidx, (Data, Name) in enumerate(zip(InputList, NameList)):
    print('  生成中({}/{}):{}'.format(musicidx+1, len(NameList), Name))
    #　初期入力
    ChordList, Pitch, Duration, Tick = copy.deepcopy(Data[0]), Data[1][0], Data[2][0], 0
    #　関数
    DataSave = SetMusic.DataSave(Logpath)   # 木探索によって生成する音楽を保存する関数
    InputChord = SetChord.InputChord()      # Tick数によって入力するコードとコード進行の値を出力する関数
    #　関数設定
    DataSave.SetChord(ChordList)
    InputChord.SetChord(ChordList,
                        StartChordNum=StartChordNum, 
                        EndChordNum=EndChordNum)
    DataSelect.SetChordDef(InputChord)

    measure = InputChord.measure                          # 生成する楽曲の小節数
    TreeSearch = Tree.TreeSearch(model, InputChord, Data2Tensor, DataSelect, DataSave, 
                                 measure, Name, Outpath, 
                                 OutputExtension=generate_args.OutputExtension,
                                 Tempo=generate_args.Tempo)  # 木探索による生成の設定
    Generate_root = TreeSearch.generate(
        Pitch=Pitch, Duration=Duration, Offset=Tick)      # 木探索による生成
    # for pre, fill, node in RenderTree(Generate_root):
    #     print("%s%s" % (pre, node.name), node.Note)
    #DotExporter(Generate_root).to_picture('{}/CheakTree_{}.png'.format(OutPath, Name))
    


     
