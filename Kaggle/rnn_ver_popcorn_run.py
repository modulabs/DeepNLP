from torchtext import data
import nltk
from bs4 import BeautifulSoup
from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable

def tokenize(text):
    pure_text = BeautifulSoup(text,"html5lib").get_text()
    tokenized = nltk.word_tokenize(pure_text)
    return tokenized

TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True,
                  batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False)

print('Loading dataset...')
train = data.TabularDataset(
            path='./src/labeledTrainData.csv', format='csv',
            skip_header=True, fields=[('Label', LABEL), ('Text', TEXT)])

print('Preparing vocabulary...')
TEXT.build_vocab(train, vectors="glove.6B.100d")

vocab = TEXT.vocab

num_epochs = 1
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 150
num_layers = 2
embedding_matrix = vocab.vectors

class SentimentAnalyzer(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 embedding_matrix):
        super(SentimentAnalyzer, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cuda = False

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, 2)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        embed_vectors = self.embedding(inputs)

        hidden_state = self.init_hidden(batch_size)
        hidden_vectors, _ = self.rnn(embed_vectors, hidden_state)
        logits = self.linear(hidden_vectors[:, -1])

        return logits

    def set_cuda(self):
        self.cuda = True

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_dim))
        if self.cuda:
            return hidden.cuda()
        return hidden

analyzer = SentimentAnalyzer(vocab_size,
                         embedding_dim,
                         hidden_dim,
                         num_layers,
                         embedding_matrix)

print('Setting train iteration...')
if torch.cuda.is_available():
    analyzer.set_cuda()
    analyzer.cuda()
    train_iter = data.BucketIterator(dataset=train, batch_size=16,
                                 sort_key=lambda x: len(x.Text),
                                 device=0, sort_within_batch=True,
                                 repeat=False)
else:
    train_iter = data.BucketIterator(dataset=train, batch_size=16,
                                 sort_key=lambda x: len(x.Text),
                                 device=-1, sort_within_batch=True,
                                 repeat=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, analyzer.parameters()))

print('Now start train!')
for epoch in range(num_epochs):
    for i, data in enumerate(tqdm(train_iter)):
        text, labels = data.Text, data.Label

        optimizer.zero_grad()
        predicts = analyzer(data.Text)
        loss = criterion(predicts, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
                print ('Loss: %.4f'
                       %(loss.data[0]))
