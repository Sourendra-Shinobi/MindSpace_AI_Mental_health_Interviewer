import pandas as pd

train = pd.read_csv('configs/dataset/train_split_Depression_AVEC2017.csv')
dev = pd.read_csv('configs/dataset/dev_split_Depression_AVEC2017.csv')
test = pd.read_csv('configs/dataset/full_test_split.csv')

train_ids = set(train['Participant_ID'].tolist())
dev_ids = set(dev['Participant_ID'].tolist())
test_ids = set(test['Participant_ID'].tolist())

audio_ids = {301,302,303,304,305,306,307,309,312,313,314,318,320,321,323,325,486}

print('IDs with AUDIO + Labels:')
all_data = []
for pid in sorted(audio_ids):
    if pid in train_ids:
        row = train[train['Participant_ID']==pid].iloc[0]
        phq = row['PHQ8_Score']; binary = row['PHQ8_Binary']; split = 'train'
    elif pid in dev_ids:
        row = dev[dev['Participant_ID']==pid].iloc[0]
        phq = row['PHQ8_Score']; binary = row['PHQ8_Binary']; split = 'dev'
    elif pid in test_ids:
        row = test[test['Participant_ID']==pid].iloc[0]
        phq = row['PHQ_Score']; binary = row['PHQ_Binary']; split = 'test'
    else:
        phq = '?'; binary = '?'; split = 'UNKNOWN'
    
    print(f'  {pid}: split={split:5s}  PHQ8={phq}  Depressed(PHQ>=10)={binary}')
    all_data.append({'pid': pid, 'split': split, 'phq8': phq, 'binary': binary})

print(f'\nTotal with audio: {len(audio_ids)}')
print(f'Depressed (PHQ>=10): {sum(1 for d in all_data if d["binary"]==1)}')
print(f'Not depressed:       {sum(1 for d in all_data if d["binary"]==0)}')
print(f'\nPHQ score distribution: {sorted([d["phq8"] for d in all_data])}')
