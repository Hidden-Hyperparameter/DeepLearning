import torch
import torchaudio
import pickle
from torch.utils.data import Dataset,DataLoader

class AISHELL_3:
    """
    A processed and REDUCTED version of AISHELL-3 dataset (https://paperswithcode.com/dataset/aishell-3).
    (Note: this dataset is much smaller than the original dataset. Moreover, the audio quality is downsampled to around 11kHz.)
    The dataset contains 10 hours of speech, in mandarin Chinese.

    This class only accept a `.pkl` file with given format: 
    - list of dict
    - each is: 
        
        >>> {
        ...     'audio': torch.Tensor, # loaded using torchaudio from wav
        ...     'sample_rate': int,
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

    def __init__(self,batch_size=2):
        print('Loading training dataset. This may take a while...')
        self.train_dataset = AISHELL3_Dataset(pickle.load(open('./data/aishell-3/train_data.pkl','rb')))
        print('Training dataset loaded.')
        self.valid_dataset = AISHELL3_Dataset(pickle.load(open('./data/aishell-3/test_data.pkl','rb')))
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
            'source': source,
            'human_readable': obj['human_readable']
        }
    
    def collate_fn(self, batch):
        audio = [x['audio'] for x in batch]
        labels = [torch.tensor(x['labels'],dtype=torch.long) for x in batch]
        return {
            'audio':torch.nn.utils.rnn.pad_sequence(audio,batch_first=True,padding_value=self.PAD_IDX),
            'labels':labels
        }
    
if __name__ == '__main__':
    a = AISHELL_3()
    valid_dl = a.valid_loader
    print(next(iter(valid_dl)))