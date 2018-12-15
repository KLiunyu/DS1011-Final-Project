from __future__ import unicode_literals, print_function, division
import numpy as np
import torch
from torch.utils.data import Dataset

from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import sacrebleu

from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

SOS_token = 1
EOS_token = 2

MAX_LENGTH = 100


class MTDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, input_lang, output_lang, pairs):
        """
        @param candiate_list: list of candidate sentence
        @param reference_list: list of reference sentence

        """
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.candidate_list = [indexesFromSentence(self.input_lang, pair[0]) for pair in pairs]
        self.reference_list = [indexesFromSentence(self.output_lang, pair[1]) for pair in pairs]
#         self.candidate_list = [pairs[i][0] for i in range(len(self.pairs))]
#         self.reference_list = [pairs[i][1] for i in range(len(self.pairs))]
        assert (len(self.candidate_list) == len(self.reference_list))

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        candidate_idx = self.candidate_list[key][:MAX_LENGTH]
        reference_idx = self.reference_list[key][:MAX_LENGTH]
#         candidate_idx.append(EOS_IDX)
#         reference_idx.append(EOS_IDX)
        candidate_idx = candidate_idx + [EOS_IDX]
        reference_idx = reference_idx + [EOS_IDX]
        return [candidate_idx, len(candidate_idx), reference_idx, len(reference_idx)]
    

def MT_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    candidate_list = []
    reference_list = []
    candidate_length_list = []
    reference_length_list = []
    for datum in batch:
        candidate_length_list.append(datum[1])
        reference_length_list.append(datum[3])
    # padding
    MAX_LENGTH = [max(candidate_length_list), max(reference_length_list)]
    for datum in batch:
        padded_vec_1 = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_LENGTH[0]-datum[1])), 
                                mode="constant", constant_values=0)
        candidate_list.append(padded_vec_1)
        
        padded_vec_2 = np.pad(np.array(datum[2]), 
                                pad_width=((0,MAX_LENGTH[1]-datum[3])), 
                                mode="constant", constant_values=0)
        reference_list.append(padded_vec_2)
    
    sorted_order = np.argsort(candidate_length_list)[::-1]
    candidate_list, candidate_length_list = np.array(candidate_list)[sorted_order], np.array(candidate_length_list)[sorted_order]
    reference_list, reference_length_list = np.array(reference_list)[sorted_order], np.array(reference_length_list)[sorted_order]
    
    return [torch.from_numpy(np.array(candidate_list)), torch.LongTensor(candidate_length_list), 
            torch.from_numpy(np.array(reference_list)), torch.LongTensor(reference_length_list)]


## Some of the followng language processing script was modified on the base of the lab code.

class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"UNK":3}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4  # Count PAD, SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1  

    def trim_sentences(self, min_count):
        if self.trimmed: 
            return
        self.trimmed = True
        
        words_keep = []
        
        for i, j in self.word2count.items():
            if j >= min_count:
                words_keep.append(i)

        print('keep_words %s / %s = %.3f' % (
            len(words_keep), len(self.word2index), len(words_keep) / len(self.word2index)
        ))


        self.word2index = {"UNK":3}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4 
        
        for word in words_keep:
            self.addWord(word)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString_en(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"&apos", r"", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

from zhon.hanzi import punctuation
def normalizeString_zh(s):
    punc = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·.'
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([。！？])", r" \1", s)
    s = re.sub(r"[0-9]", r" ", s)
    s = re.sub(r"[%s]+" %punc, r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines1 = open(lang1, encoding='utf-8').\
        read().strip().split('\n')
    
    lines2 = open(lang2, encoding='utf-8').\
        read().strip().split('\n')
    
    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [None] * len(lines1)
    print(len(pairs))
    for i in range(len(lines1)):
        pairs[i] = [normalizeString_en(lines1[i]), normalizeString_zh(lines2[i])]
        
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    print('start to read')
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, input_seqs, input_lengths, batch_size, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.numpy(), batch_first = True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        hidden = hidden[:self.n_layers, :, :] + hidden[self.n_layers:,:,:]
        
        return outputs, hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(self.n_layers*2, batch_size, self.hidden_size, device=device)

    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, batch_size):
        #import pdb;pdb.set_trace()
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_size = encoder_outputs.size()[0]
        attn_weights = attn_weights[:,:attn_size]
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                     encoder_outputs.transpose(0,1))

        output = torch.cat((embedded[0], attn_applied[:,0,:]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    out = []
    for word in sentence.split(' '):
        if word in lang.word2index.keys():
            out.append(lang.word2index[word])
        else:
            out.append(lang.word2index['UNK'])
    return out


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 1


def train(input_tensor, input_lengths, target_tensor, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_tensor = input_tensor.cuda()
    target_tensor = target_tensor.cuda()

    #input_length = input_tensor.size(0)
    target_length = max(target_lengths.numpy())
    batch_size = input_tensor.size(0)

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    encoder_hidden = encoder.initHidden(batch_size)
    encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths, batch_size, encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device)

    decoder_hidden = encoder_hidden
    
    #import pdb;pdb.set_trace()

    use_teacher_forcing = True if random.random() <= teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, batch_size)
            loss += criterion(decoder_output, target_tensor[:,di])
            decoder_input = target_tensor[:,di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


teacher_forcing_ratio = 1

def train_wo(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max(target_length)):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

## Some of the followng utility function script was modified on the base of the lab code.

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

                                                      
def mapback(reference):

    words = []
    
    for i in range(reference.size(0)):
        line = []
        for j in range(reference.size(1)):
            if int(reference[i,j].item()) == 2:
                break
            else:
                line.append(output_lang.index2word[int(reference[i,j].item())])
        
        line = ' '.join(line)
        words.append(line)
      
    return words
  
def evaluate(encoder, decoder, candidate, length_1, reference, length_2, max_length):
  
   with torch.no_grad():
                                                      
    batch_size = candidate.size(0)
    candidate =  candidate.cuda()
    reference = reference.cuda()
    encoder_outputs, encoder_hidden = encoder(candidate, length_1, None)
    decoder_input = torch.tensor([[SOS_token]*batch_size], device = device)
    decoder_hidden = encoder_hidden
    batch_size = candidate.size(0)
    
    decoded_words = torch.ones([batch_size, max_length])
    
    for di in range(max_length):

        try:

            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, batch_size)
        except:

            import pdb; pdb.set_trace()

        topv, topi = decoder_output.topk(1)
        decoded_words[:,di] = topi.squeeze()
        decoder_input = topi.squeeze().detach().cuda()
        #decoder_input = topi.squeeze().detach()
        words = mapback(decoded_words) 
      
        
    return words
    
def evaluatebleu(encoder1, decoder1, loader, with_o = True):
  
    score = 0
    output_words = []
    true_words = []
    for i, (candidate, length_1, reference, length_2) in enumerate(loader):
        #print(i)
        if with_o == True:
            max_length = max(length_2).item()
            output_words += evaluate(encoder1, decoder1, candidate, length_1, reference, length_2,max_length)
        else:
            output_words, attentions = evaluate(encoder, decoder, pair[0])
        true_words += mapback(reference)
        
    score = sacrebleu.corpus_bleu(output_words,[true_words])
    
    print(output_words[0])
    print(true_words[0])
        
        
    return (score, output_words, true_words)
                                                      
def trainIters(encoder, decoder, n_iters, train_loader, val_loader, print_every=1000, plot_every=100, learning_rate=0.0005, with_o = True):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter in range(n_iters):
        for i, (data, data_lengths, ref, ref_lengths) in enumerate(train_loader):
            #import pdb;pdb.set_trace()
            input_tensor = data
            target_tensor = ref        
        
            if with_o == True:
                loss = train_wo(input_tensor, data_lengths, target_tensor, ref_lengths, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            else:
                loss = train(input_tensor, data_lengths, target_tensor, ref_lengths, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
        
        
        
            if (i+1) % print_every == 0:
                #import pdb;pdb.set_trace()
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s epoch, %s (%d %d%%) %.4f' % ( (iter+1), timeSince(start, i / len(train_loader)),
                                         (i+1), (i+1) / len(train_loader) * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        
        #showPlot(plot_losses)
        try:
            fig_path = 'loss_%s.png' %(iter+1)
            plt.savefig(fig_pth)
        except:
            pass
        s, output_words, true_words = evaluatebleu(encoder1, attn_decoder1, val_loader)
        print(s)                                              
        path_en = "/scratch/cl4055/DS1011_model/encoder1_attn_chi_0.0005_t_%s.pth" %(iter+1)
        path_de = "/scratch/cl4055/DS1011_model/decoder1_attn_chi_0.0005_t_%s.pth" %(iter+1)
        torch.save(encoder1.state_dict(), path_en)
        torch.save(attn_decoder1.state_dict(), path_de)
    return plot_losses
                                                      
                                                     
print('prepare data')
input_lang, output_lang, pairs = prepareData('train.tok.en', 'train.tok.zh', True)
val_input_lang, val_output_lang, val_pairs = prepareData('dev.tok.en', 'dev.tok.zh', True)
print(random.choice(pairs))
BATCH_SIZE = 32
hidden_size = 512

MIN_COUNT = 2
input_lang.trim_sentences(MIN_COUNT)
output_lang.trim_sentences(MIN_COUNT)

train_dataset = MTDataset(input_lang, output_lang, pairs)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_size=BATCH_SIZE,
                                          collate_fn=MT_collate_func,
                                          shuffle=True)
                                                      

val_dataset = MTDataset(input_lang, output_lang, val_pairs)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                          batch_size=BATCH_SIZE,
                                          collate_fn=MT_collate_func,
                                          shuffle=True)                                                      
                                                   
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
train_loss = trainIters(encoder1, attn_decoder1, 10, train_loader, val_loader, print_every= 500, with_o = False)
torch.save(encoder1.state_dict(), "/scratch/cl4055/DS1011_model/encoder1_attn_chi_0.0005.pth")
torch.save(attn_decoder1.state_dict(), "/scratch/cl4055/DS1011_model/decoder1_attn_chi_0.0005.pth")
#evaluateRandomly(encoder1, attn_decoder1, with_o = False)