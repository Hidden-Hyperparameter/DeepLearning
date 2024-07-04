import torch
import torchaudio
import pickle
from torch.utils.data import Dataset,DataLoader

def saveaudio(tensor,sample_rate,path):
    if len(tensor.shape)==1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(path,tensor,sample_rate,format='wav')

class AISHELL_3:
    """
    A processed and REDUCTED version of AISHELL-3 dataset (https://paperswithcode.com/dataset/aishell-3).
    (Note: this dataset is much smaller than the original dataset. Moreover, the audio quality is downsampled to around 11025 Hz.)
    The dataset contains 10 hours of speech, in mandarin Chinese.

    This class only accept a `.pkl` file with given format: 
    - list of dict
    - each is: 
        
        >>> {
        ...     'audio': torch.Tensor, # loaded using torchaudio from wav, shape:
        ...     'sample_rate': int, # optioal, default to 11025
        ...     'labels':{
        ...             'age group':int # in [0,1,2,3], corresponds to [A:< 14, B:14 - 25, C:26 - 40, D:> 41.]
        ...             'gender':int # in [0,1], corresponds to ['male','female']
        ...             'accent':int # in [0,1,2], corresponds to ['north','south','others']
        ...     },
        ...     'source': list[int], # tokenized sentence, start with 0 and end with 1. constants are: '^': 0; '$': 1; '%'(delimeter): 2; '<pad>': 3. NOTICE: must remove "%" when training and generation, since it is just for human to reference, but the information can't be revealed to model.
        ...     'human_readable': str # human readable sentence with delimeter
        ... }

    The processing code can be found at `additional_utils/aishell-3.py`
    """

    BOS_IDX = 0
    EOS_IDX = 1
    SEP_IDX = 2
    PAD_IDX = 3
    PHONE_DICTIONARY_SIZE = 1930 # phone: sound units (pinyin)

    def __init__(self,batch_size=2):
        print('Loading training dataset. This may take a while...')
        self.train_dataset = AISHELL3_Dataset(pickle.load(open('../data/aishell-3/train_data.pkl','rb')))
        print('Training dataset loaded.')
        self.valid_dataset = AISHELL3_Dataset(pickle.load(open('../data/aishell-3/test_data.pkl','rb')))
        self.train_loader = DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True,collate_fn=self.train_dataset.collate_fn)
        self.valid_loader = DataLoader(self.valid_dataset,batch_size=batch_size,shuffle=True,collate_fn=self.valid_dataset.collate_fn)

class AISHELL3_Dataset(Dataset):

    BOS_IDX = 0
    EOS_IDX = 1
    SEP_IDX = 2
    PAD_IDX = 3

    def __init__(self, data):
        self.data = data # list of dict
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        # process object source
        source = obj['source']
        source = [c for c in source if c != 2] # remove delimeter
        return {
            'audio':  obj['audio'],
            'labels': obj['labels']['age group']*6+obj['labels']['gender']*3+obj['labels']['accent'],
            'source': torch.tensor(source,dtype=torch.int),
            'human_readable': obj['human_readable']
        }
    
    def collate_fn(self, batch):
        audio = [x['audio'] for x in batch]
        labels = [torch.tensor(x['labels'],dtype=torch.long) for x in batch]
        return {
            'audios':torch.nn.utils.rnn.pad_sequence(audio,batch_first=True,padding_value=0),
            'sources':torch.nn.utils.rnn.pad_sequence([x['source'] for x in batch],batch_first=True,padding_value=self.PAD_IDX),
            'labels':labels
        }
    
class MusicGenres:
    
    """
    A processed and REDUCTED version of music-genre dataset (https://huggingface.co/datasets/lewtun/music_genres).
    (Note: this dataset is much smaller than the original dataset. Moreover, the audio quality is downsampled to around 11025 Hz.)
    The dataset contains 20 hours of musics.

    This class only accept a `.pkl` file with given format: 
    - list of dict
    - each is: 
        
        >>> {
        ...     'audio': torch.Tensor, # loaded using torchaudio from wav
        ...     'label': int, # the music genre. See below for genre map.
        ... }

    The processing code can be found at `additional_utils/music-genre.py`
    """
    GENRE_MAP = {
        -1: 'Unknown', 
        0: 'Electronic', 
        1: 'Rock', 
        2: 'Punk', 
        3: 'Experimental', 
        4: 'Hip-Hop', 
        5: 'Folk', 
        6: 'Chiptune / Glitch', 
        7: 'Instrumental', 
        8: 'Pop', 
        9: 'International', 
        10: 'Ambient Electronic', 
        11: 'Classical', 
        12: 'Old-Time / Historic', 
        13: 'Jazz', 
        14: 'Country', 
        15: 'Soul-RnB', 
        16: 'Spoken', 
        17: 'Blues', 
        18: 'Easy Listening'
    }

    def __init__(self,batch_size=8):
        print('Loading training dataset. This may take a while...')
        # self.train_dataset = MusicGenresDataset(pickle.load(open('../data/music_genres/train_data.pkl','rb')))
        print('Training dataset loaded.')
        self.valid_dataset = MusicGenresDataset(pickle.load(open('../data/music_genres/validation_data.pkl','rb')))
        # self.train_loader = DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True,collate_fn=self.train_dataset.collate_fn)
        self.train_loader = 1
        self.valid_loader = DataLoader(self.valid_dataset,batch_size=batch_size,shuffle=True,collate_fn=self.valid_dataset.collate_fn)

class MusicGenresDataset(Dataset):

    def __init__(self, data):
        self.data = data # list of dict
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, batch):
        unified_channel = max([
            (len([x['audio'] for x in batch if x['audio'].shape[0]==1]),1),
            (len([x['audio'] for x in batch if x['audio'].shape[0]==2]),2)
        ])[-1] # as the data may have different channels, we unify the channel to the most common one.
        batch = [x for x in batch if x['audio'].shape[0]==unified_channel]
        audio = [x['audio'].transpose(0,1) for x in batch] # each of shape: [channel,leng]
        labels = [x['label']+1 for x in batch] # make sure label is >0
        return {
            'audios':torch.nn.utils.rnn.pad_sequence(audio,batch_first=True,padding_value=0).transpose(-1,-2),
            'labels':torch.tensor(labels,dtype=torch.int),
        }
    

if __name__ == '__main__':
    # a = AISHELL_3()
    a = MusicGenres()
    valid_dl = a.valid_loader
    a = (next(iter(valid_dl)))
    x,y = a['audios'],a['labels']
    print(x.shape,y.shape)