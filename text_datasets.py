import datasets
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader,Dataset
import re,json,os
from tqdm import tqdm

class Dictionary:
    """dictionary:
    1. load dataset: create a map, from word to index; process the dataset into a list of indices
    2. encode line: convert a line to indices
    """

    def __init__(self,max_size=50000):
        self.vocab = {
            '<pad>':float('inf'),
            '<bos>':float('inf'),
            '<eos>':float('inf'),
            '<unk>':float('inf')
        }
        self.max_size = max_size
        self.init_finished = False
    
    def tokenize(self,sentence):
        raise NotImplementedError()

    def add(self,sentence):
        tokens = self.tokenize(sentence)
        for token in tokens:
            if token in self.vocab:
                self.vocab[token] += 1
            else:
                self.vocab[token] = 1

    def encode_ln(self,line):
        assert self.init_finished
        main = [self.vocab.get(word,self.unk) for word in self.tokenize(line)]
        return [self.bos]+main+[self.eos]
    
    def finish_init(self):
        print('init finished with {} words'.format(len(self.vocab)))
        self.init_finished = True
        self.reverse_vocab = {idx:word for word,idx in self.vocab.items()}

    def save(self,path=None):
        if path is None:
            if not hasattr(self,'save_path'):
                raise ValueError('save path not specified')
            path = self.save_path
        l = [(times,word) for word,times in self.vocab.items()]
        l = sorted(l[:self.max_size],reverse=True)
        self.vocab = {word:idx for idx,(_,word) in enumerate(l)}
        self.finish_init()
        json.dump(self.vocab, open(path, 'w'))

    def load(self,path=None):
        self.vocab = json.load(open(path, 'r'))
        if len(self.vocab)>self.max_size:
            self.vocab = {word:ind for word,ind in self.vocab if ind<self.max_size}
            assert all([x in self.vocab for x in ['<pad>','<bos>','<eos>','<unk>']]),'Invalid dictionary file'
        self.finish_init()
    
    def decode_ln(self,indices):
        assert self.init_finished
        if isinstance(indices,torch.Tensor):
            assert len(indices.shape)==1,'Does not accept batch input.'
            indices = indices.long().detach().cpu().tolist()
        return ' '.join(self.reverse_vocab.get(idx,self.unk) for idx in indices)

    @property
    def bos(self): assert self.init_finished; return self.vocab['<bos>']
    @property
    def eos(self): assert self.init_finished; return self.vocab['<eos>']
    @property
    def pad(self): assert self.init_finished; return self.vocab['<pad>']
    @property
    def unk(self): assert self.init_finished; return self.vocab['<unk>']

class ZH_Dictionary(Dictionary):

    def __init__(self, max_size=50000):
        super().__init__(max_size)
        self.save_path = f'./data/zh_dict_max{max_size}.json'

    def tokenize(self,sentence):
        return re.findall(r'([^\da-z]|[\da-z]+)', sentence.lower())
    
    def save(self,path=None):
        if path is None:
            if not hasattr(self,'save_path'):
                raise ValueError('save path not specified')
            path = self.save_path
        l = [(times,word) for word,times in self.vocab.items()]
        l = sorted(l[:self.max_size],reverse=True)
        self.vocab = {word:idx for idx,(_,word) in enumerate(l)}
        self.finish_init()
        json.dump(self.vocab, open(path, 'w',encoding='utf-8'),ensure_ascii=False)

    def load(self,path=None):
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.finish_init()

    def decode_ln(self,indices):
        assert self.init_finished
        if isinstance(indices,torch.Tensor):
            assert len(indices.shape)==1,'Does not accept batch input.'
            indices = indices.long().detach().cpu().tolist()
        return ''.join(self.reverse_vocab.get(idx,self.unk) for idx in indices)

class EN_Dictionary(Dictionary):

    def __init__(self, max_size=50000):
        super().__init__(max_size)
        self.save_path = f'./data/en_dict_max{max_size}.json'

    def tokenize(self,sentence):
        return re.findall(r'\b\w+\b|[^\w\s]', sentence.lower())

class WMT19Dataset(Dataset):

    def __init__(self,dataset:datasets.arrow_dataset.Dataset,load=False,en_dict = None,zh_dict = None):
        self.dataset = dataset
        if load:
            print('Using preloaded dictionary.')
            en_dict = EN_Dictionary()
            en_dict.load('./data/en_dict_max50000.json')
            self.en_dict = en_dict
            zh_dict = ZH_Dictionary()
            zh_dict.load('./data/zh_dict_max50000.json')
            self.zh_dict = zh_dict
            return
        if en_dict is not None and zh_dict is not None:
            self.en_dict = en_dict
            self.zh_dict = zh_dict
            return
        
        self.en_dict = EN_Dictionary()
        self.zh_dict = ZH_Dictionary()
        def pre_process(one):
            self.en_dict.add(' '.join(x['en'] for x in one['translation']))
            self.zh_dict.add(''.join(x['zh'] for x in one['translation']))
        dataset.map(pre_process, batched=True)
        self.en_dict.save()
        self.zh_dict.save()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw = self.dataset[idx]
        print(raw)
        en = self.en_dict.encode_ln(raw['translation']['en'])
        zh = self.zh_dict.encode_ln(raw['translation']['zh'])
        return en,zh

class WMT19DataLoader(DataLoader):

    def __init__(self,dataset:Dataset,batch_size=32,shuffle=True):
        self.dataset = dataset
        super().__init__(self.dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=self.collate_fn)

    def collate_fn(self,batch):
        en,zh = zip(*batch)
        en_len = [len(x) for x in en]
        zh_len = [len(x) for x in zh]
        en = [x+[self.dataset.en_dict.pad]*(max(en_len)-len(x)) for x in en]
        zh = [x+[self.dataset.zh_dict.pad]*(max(zh_len)-len(x)) for x in zh]
        en = torch.tensor(en)
        zh = torch.tensor(zh)
        return en,zh


class WMT19:

    def __init__(self,batch_size=32) -> None:
        assert os.path.exists('./data/wmt19'),'Download at https://huggingface.co/datasets/wmt/wmt19/tree/main/zh-en'
        data_files = {
            "train": ["./data/wmt19/train-000{:02d}-of-00013.parquet".format(i) for i in range(13)],
            "validation": "./data/wmt19/validation-00000-of-00001.parquet"
        }
        wmt19 = load_dataset('parquet', data_files=data_files)
        print('WMT19 loaded. train dataset len:', len(wmt19['train']))
        # raise NotImplementedError()
        self.train_dataset = WMT19Dataset(wmt19['train'],load=True)
        self.valid_dataset = WMT19Dataset(wmt19['validation'],en_dict=self.train_dataset.en_dict,zh_dict=self.train_dataset.zh_dict)
        self.train_dataloader = WMT19DataLoader(self.train_dataset,batch_size=batch_size)
        self.valid_dataloader = WMT19DataLoader(self.valid_dataset,batch_size=batch_size)

# if __name__ == '__main__':
#     wmt19 = WMT19()
#     train_dl = wmt19.train_dataloader
#     valid_dl = wmt19.valid_dataloader
#     en,zh = next(iter(train_dl))