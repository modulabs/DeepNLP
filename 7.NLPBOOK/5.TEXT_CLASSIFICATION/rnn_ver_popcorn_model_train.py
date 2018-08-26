from torchtext import data
import nltk
from bs4 import BeautifulSoup
from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

def tokenize(text):
    pure_text = BeautifulSoup(text,"html5lib").get_text()
    tokenized = nltk.word_tokenize(pure_text)
    return tokenized

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    target = target.long()

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target = target.view(1, -1)
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class SentimentAnalyzer(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 embedding_matrix=None):
        super(SentimentAnalyzer, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_cuda = False

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
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

        predicts = F.softmax(logits)
        log_logits = F.log_softmax(logits, dim=-1)

        return predicts, log_logits

    def set_cuda(self):
        self.use_cuda = True

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_dim))
        if self.use_cuda:
            return hidden.cuda()
        return hidden

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True,
                  batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False)

print('Loading data...')
train, dev = data.TabularDataset.splits(
              path='./src', train='labeledTrainData.csv', validation='labeledDevData.csv', format='csv',
              skip_header=True, fields=[('Label', LABEL), ('Text', TEXT)])

print('Preparing vocabulary...')
TEXT.build_vocab(train, vectors="glove.6B.100d")

vocab = TEXT.vocab

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 150
num_layers = 2
embedding_matrix = vocab.vectors

hyper_params = {'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers}

print('Model initiate!')
analyzer = SentimentAnalyzer(vocab_size,
                             embedding_dim,
                             hidden_dim,
                             num_layers,
                             embedding_matrix)

if torch.cuda.is_available():
    analyzer.set_cuda()
    analyzer.cuda()
    train_iter, dev_iter = data.BucketIterator.splits((train, dev), batch_size=16,
                                 sort_key=lambda x: len(x.Text),
                                 device=0, sort_within_batch=True,
                                 repeat=False)
else:
    train_iter, dev_iter = data.BucketIterator.splits((train, dev), batch_size=16,
                                 sort_key=lambda x: len(x.Text),
                                 device=-1, sort_within_batch=True,
                                 repeat=False)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, analyzer.parameters()))


print('Train start!')
num_epochs = 5
analyzer.train()
for epoch in range(num_epochs):
    acc = AverageMeter()
    _loss = AverageMeter()
    for i, data in enumerate(tqdm(train_iter)):
        text, labels = data.Text, data.Label

        optimizer.zero_grad()
        predicts, log_logits = analyzer(text)
        loss = criterion(log_logits, labels)
        loss.backward()
        optimizer.step()

        temp_acc = accuracy(predicts, labels)
        acc.update(temp_acc[0], text.size(0))
        _loss.update(loss.data[0], text.size(0))
        if (i+1) % 100 == 0:
            print ('Epoch: %d Step: %d Loss: %.4f Accuracy: %.2f'%(epoch, i+1, _loss.avg, acc.avg))

            val_acc = AverageMeter()
            analyzer.eval()
            for i, data in enumerate(tqdm(dev_iter)):
                text, labels = data.Text, data.Label
                predicts, _ = analyzer(text)
                temp_acc = accuracy(predicts, labels)
                val_acc.update(temp_acc[0], text.size(0))
            print ('Validation Accuracy: %.2f'%(acc.avg))
            analyzer.train()

    file_path = 'save_model_%d.pth.tar' % (epoch + 1)
    torch.save({'epoch': epoch + 1,
                'state_dict': analyzer.cpu().state_dict(),
                'accuracy': acc.avg,
                'optimizer' : optimizer.cpu().state_dict(),
                'vocab': vocab.stoi,
                'hyper_params': hyper_params}, file_path)
    analyzer.cuda()
    optimizer.cuda()
