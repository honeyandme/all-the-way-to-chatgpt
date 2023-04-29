import os
import torch
from torch.utils.data import Dataset,DataLoader
def read_data(path,num=None):
    with open(path,encoding='utf-8') as f:
        all_data = f.read().split('\n\n')
    if num is not None:
        return all_data[:-1][:num]
    return all_data[:-1]
def build_word_2_index(path):
    with open(path, encoding='utf-8') as f:
        index_2_word = f.read().split('\n')
    word_2_index = {k:v for v,k in enumerate(index_2_word)}
    return word_2_index,index_2_word

class G_dataset(Dataset):
    def __init__(self,all_data,word_2_index):
        self.all_data = all_data
        self.word_2_index = word_2_index
    def __getitem__(self, x):
        data = self.all_data[x].split('\n')
        text_idx = []
        for d in data:
            text_idx.extend([word_2_index[i] for i in d])
            text_idx.append(2)
        input_idx = text_idx[:-1]
        label_idx = text_idx[1:]

        assert len(input_idx) == len(label_idx) ,'sb.长度不一样'
        return input_idx,label_idx,len(input_idx)
    def process_data(self,data):
        batch_input,batch_label,batch_len = zip(*data)
        batch_max_len = max(batch_len)
        batch_new_input,batch_new_label = [],[]
        for input_idx,label_idx in zip(batch_input,batch_label):
            batch_new_input.append(input_idx+[0]*(batch_max_len-len(input_idx)))
            batch_new_label.append(label_idx+[0]*(batch_max_len-len(label_idx)))

        return torch.tensor(batch_new_input),torch.tensor(batch_new_label)
    def __len__(self):
        return len(self.all_data)
if __name__ == '__main__':
    all_data = read_data(os.path.join('data','train.txt'),200)
    word_2_index,index_2_word = build_word_2_index(os.path.join('data','vocab.txt'))

    batch_size = 3
    epoch = 1

    train_dataset = G_dataset(all_data,word_2_index)
    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size=batch_size,collate_fn=train_dataset.process_data)

    for e in range(epoch):
        for x,y in train_dataloader:
            print()