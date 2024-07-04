import os
import torch,pickle
from tqdm import tqdm
import torchaudio
from datasets import load_dataset

root = './data'

def process_1():
    assert os.path.exists(f'{root}/music_genres'),'Please download the dataset at https://huggingface.co/datasets/lewtun/music_genres/tree/main/data'

    data_files = {
        "train": [f"{root}/music_genres/train-000{i:02d}-of-00016.parquet" for i in range(2)],
        "validation": f"{root}/music_genres/test-00000-of-00004.parquet"
    }
    data = load_dataset('parquet', data_files=data_files)
    print('loaded. train dataset len:', len(data['train']),'valid dataset len:',len(data['validation']))
    outs = {
        'train':[],
        'validation':[]
    }
    genre_map = dict()
    for task in outs:
        i = 0
        for item in tqdm(data[task]):
            i += 1
            if task == 'validation' and i > 400:
                break
            bts = item['audio']['bytes']
            open('tmp.wav','wb').write(bts)
            tensor,sample = torchaudio.load('tmp.wav')
            # downsample
            # print('downsample...')
            tensor = torchaudio.transforms.Resample(orig_freq=sample,new_freq=11025)(tensor)
            sample = 11025
            # if too long, clip tensor to 10 seconds
            tensors = []
            i = 1
            while tensor.shape[1] > 10*sample:
                cutted = tensor[:,:10*sample]
                tensor = tensor[:,10*sample:]
                tensors.append(cutted.clone())
                i += 1
            tensors.append(tensor)
            # if i > 1:
            #     print('[INFO] clip to',i,'tensors')
            for tensor in tensors:
                outs[task].append({
                    'audio':tensor,
                    'label':item['genre_id']
                })
            genre_map.update({item['genre_id']:item['genre']})
        pickle.dump(outs[task],open(f'{root}/music_genres/{task}_data.pkl','wb'))

    print('Genre map:',genre_map)

def cut_dataset(dataset,out_path):
    print('Loading dataset...')
    data_1 = pickle.load(open(dataset,'rb'))
    print('Loaded. Length:',len(data_1))
    data_1 = data_1[:len(data_1)//10]
    out = []
    for item in tqdm(data_1):
        tensor = item['audio']
        tensors = []
        while tensor.shape[1] > 11025*5:
            cutted = tensor[:,:11025*5]
            tensor = tensor[:,11025*5:]
            tensors.append(cutted.clone())
        tensors.append(tensor)
        out.extend([{
            'audio':x,
            'label':item['label']
        } for x in tensors])
    pickle.dump(out,open(out_path,'wb'))

if __name__ == '__main__':
    # process_1()
    cut_dataset(f'{root}/music_genres/validation_data.pkl',f'{root}/music_genres/validation_data_short.pkl')
    # cut_dataset(f'{root}/music_genres/train_data.pkl',f'{root}/music_genres/train_data_short.pkl')