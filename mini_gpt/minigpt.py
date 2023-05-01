import os
import torch
from torch import nn
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

class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len,768)
        self.token_emb = nn.Embedding(vocab_len,768)
    def forward(self,x):
        position = torch.arange(0, x.shape[1], device=x.device).reshape(1, -1)
        position = position.expand_as(x)
        x = self.token_emb(x) + self.pos_emb(position)

        return x

class Feed_Forward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_block1 = MultiHeadAttention()
        self.attention_block2 = MultiHeadAttention()
        self.feed = Feed_Forward()
    def forward(self,x):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        self.layers = [DecoderBlock() for i in range(3)]


    def forward(self,x):
        x = self.embedding(x)
        return x

class GPT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.cls = nn.Linear(768,vocab_len)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,x,y):
        x =self.decoder(x)
        return x


if __name__ == '__main__':
    all_data = read_data(os.path.join('data','train.txt'),200)
    word_2_index,index_2_word = build_word_2_index(os.path.join('data','vocab.txt'))

    vocab_len = len(word_2_index)
    batch_size = 3
    epoch = 1
    max_seq_len = 512
    lr = 0.0001


    train_dataset = G_dataset(all_data,word_2_index)
    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size=batch_size,collate_fn=train_dataset.process_data)

    model = GPT_Model()
    opt = torch.optim.Adam(model.parameters(),lr = lr)

    for e in range(epoch):
        for x,y in train_dataloader:

            loss = model(x,y)