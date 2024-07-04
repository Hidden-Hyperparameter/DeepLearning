import torch
import torchaudio
import csv,os
from tqdm import tqdm
import pickle

def wav2ten(filename):
    waveform, sample_rate = torchaudio.load(filename)
    return waveform[0][::4],sample_rate//4 # downsample

BOS_IDX = 0
EOS_IDX = 1
SEP_IDX = 2
PAD_IDX = 3

def get_dictionary():
    # tokens.txt comes from phone_set.txt and manually remove all `\t` to one space.
    result = csv.reader(open('tokens.txt', 'r'),delimiter=' ')
    dictionary = set()
    for row in result:
        dictionary.add(row[0])
    out = {c:i+5 for i,c in enumerate(dictionary)}
    out.update({
        '^': BOS_IDX,
        '$': EOS_IDX,
        '%': SEP_IDX, # delimeter
        '<pad>': PAD_IDX,
    })
    return out

def get_source_tokens_by_label():
    dictioary = get_dictionary()
    # to get label.txt: do
    # `cat label_train-set.txt | grep -E --invert-match '^#' | sort > label.txt`
    result = csv.reader(open('label.txt', 'r'),delimiter='|')
    ans = dict()
    for row in result:
        ans.update({
            row[0]:{
                'source': [BOS_IDX]+[dictioary[c] for c in row[1].split(' ')],
                'human_readable': row[2]
            }
        })
    return ans

def get_source_tokens_by_content():
    dictioary = get_dictionary()
    lines = open('content.txt').readlines()
    ans = dict()
    for line in lines:
        line = line.strip(' \n')
        if not '.wav' in line:
            continue
        audio_id,content = line.split('.wav')
        content = content.split(' ')
        chn,eng = content[::2],content[1::2]
        ans.update({
            audio_id:{
                'source': [BOS_IDX]+[dictioary[c] for c in eng],
                'human_readable': ''.join(chn)
            }
        })
    return ans

def get_emotion_labels():
    # In years, A:< 14, B:14 - 25, C:26 - 40, D:> 41.
    result = csv.reader(open('spk-info.txt', 'r'),delimiter=' ',quotechar='#')
    ans = dict()
    for row in result:
        ans.update({
            row[0]:{
                'age group':['A','B','C','D'].index(row[1]),
                'gender':['male','female'].index(row[2]),
                'accent':['north','south','others'].index(row[3]),
            }
        })
    return ans

def get_all():
    tokens = get_source_tokens_by_content()
    # tokens = get_source_tokens_by_label()
    labels = get_emotion_labels()
    dirs = os.listdir('.')
    for one in tqdm(dirs):
        if os.path.isdir(one):
            files = os.listdir(one)
            for file in files:
                if file.endswith('.wav'):
                    tensor,rate = wav2ten(os.path.join('.',one,file))
                    audio_id = file.removesuffix('.wav')
                    info = tokens[audio_id].copy()
                    info.update({
                        'audio':tensor,
                        'sample_rate':rate,
                        'labels':labels[audio_id[:-4]]
                    })
                    yield info

def saveaudio(tensor,sample_rate,path):
    if len(tensor.shape)==1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(path,tensor,sample_rate)

if __name__ == '__main__':
    # pickle.dump(list(get_all()),open('train_data.pkl','wb'))
    pickle.dump(list(get_all()),open('test_data.pkl','wb'))
    # audio,sample_rate = wav2ten('./SSB0005/SSB00050004.wav')
    # saveaudio(audio,sample_rate,'test.wav')
    # print(get_emotion_labels())