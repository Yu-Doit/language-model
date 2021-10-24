# coding: utf-8
import argparse
import time
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model import TransformerLM, LSTMLM, repackage_hidden
from data import Corpus, batchify, get_batch


parser = argparse.ArgumentParser(description = 'Transformer/LSTM Language Model')
parser.add_argument('--data', type = str, default = './data/ptb', help = 'location of the data corpus')
parser.add_argument('--model', type = str, default = 'Transformer', help = 'type of recurrent net (Transformer, LSTM)')
parser.add_argument('--emsize', type = int, default = 256, help = 'size of word embeddings')
parser.add_argument('--nhid', type = int, default = 1024, help = 'number of hidden units per layer')
parser.add_argument('--nlayers', type = int, default = 6, help = 'number of layers')
parser.add_argument('--nhead', type = int, default = 4, help = 'the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--lr', type = float, default = 1e-4, help = 'initial learning rate')
parser.add_argument('--optim', type = str, default = None, help = 'optimizer')
parser.add_argument('--clip', type = float, default = 0.25, help = 'gradient clipping')
parser.add_argument('--epochs', type = int, default = 80, help = 'upper epoch limit')
parser.add_argument('--batch_size', type = int, default = 32, metavar = 'N', help = 'batch size')
parser.add_argument('--eval_bsz', type = int, default = 32, help = 'eval batch size')
parser.add_argument('--bptt', type = int, default = 70, help = 'sequence length')
parser.add_argument('--dropout', type = float, default = 0.2, help = 'dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action = 'store_true', help = 'tie the word embedding and softmax weights')
parser.add_argument('--seed', type = int, default = 1111, help = 'random seed')
parser.add_argument('--cuda', action = 'store_true', help = 'use CUDA')
parser.add_argument('--log-interval', type = int, default = 80, metavar = 'N', help = 'report interval')
parser.add_argument('--save', type = str, default = 'model.pt', help = 'path to save the final model')
parser.add_argument('--No', type = int, default = 0, help = 'experiment No.')
parser.add_argument('--pre', type = str, default = None, help = 'pretrained chekpoint')
parser.add_argument('--dry-run', action = 'store_true', help = 'verify the code and the model')
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')

corpus = Corpus(args.data)
train_data = batchify(corpus.train, args.batch_size).to(device)
val_data = batchify(corpus.valid, args.eval_bsz).to(device)
test_data = batchify(corpus.test, args.eval_bsz).to(device)

ntokens = len(corpus.vocab)
if args.model == 'Transformer':
    model = TransformerLM(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = LSTMLM(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()
lr = args.lr
if args.optim is not None:
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = lr)

log_path = './log/logging_' + str(args.No) + '.txt'
model_path = './model/' + str(args.No) + '_' + args.save
final_model_path = './final_model/' + str(args.No) + '_' + args.save

def logging(s, path):
    print(s)
    with open(path, 'a+', encoding = 'utf8') as f:
        f.write(s + '\n')

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.vocab)

    if args.model == 'LSTM':
        hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, target = get_batch(train_data, i, args.bptt)
        
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data).view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.optim is None:
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)
        else:
            optimizer.step()
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            log_info = '| epoch {:3d} | {:5d}/{:5d} batches | lr {} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss))
            logging(log_info, log_path)
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.vocab)
    
    if args.model == 'LSTM':
        hidden = model.init_hidden(args.eval_bsz)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, target = get_batch(data_source, i, args.bptt)
            if args.model == 'Transformer':
                output = model(data).view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, target).item()

    return total_loss / (len(data_source) - 1)

best_val_loss = None

try:
    log_info = 'Model type: ' + str(args.model) + '  Dataset: ' + str(args.data) + '  Experiment {} starts.'.format(args.No)
    logging(log_info, log_path)

    log_info = 'd_model: ' + str(args.emsize) + '  d_feedforward: ' + str(args.nhid) + '  n_layer: ' + str(args.nlayers) + '  n_head: ' + str(args.nhead)
    logging(log_info, log_path)

    log_info = 'batchsize: ' + str(args.batch_size) + '  bptt: ' + str(args.bptt) + '  lr: ' + str(args.lr) + '  dropout: ' + str(args.dropout)
    logging(log_info, log_path)

    if args.pre is not None:
        with open(args.pre, 'rb') as f:
            state = torch.load(f)
        
        model.load_state_dict(state['model'])
        if args.optim is not None:
            optimizer.load_state_dict(state['optimizer'])
            
        if args.model == 'LSTM':
            model.lstm.flatten_parameters()

        log_info = 'Continue training from ckeckpoint: {}'.format(args.pre)
        logging(log_info, log_path)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        
        log_info = '-' * 89
        logging(log_info, log_path)
        
        log_info = '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
            epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss))
        logging(log_info, log_path)
        
        log_info = '-' * 89
        logging(log_info, log_path)

        if not best_val_loss or val_loss < best_val_loss:
            if args.optim is not None:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            else:
                state = {'model': model.state_dict()}
            with open(model_path, 'wb') as f:
                torch.save(state, f)
            best_val_loss = val_loss

            log_info = 'save model. valid loss {:5.2f} valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss))
            logging(log_info, log_path)
        else:
            if args.optim is None:
                lr /= 4.0
            else:
                lr /= 2.0
                for param in optimizer.param_groups:
                    param['lr'] = lr

    
    if args.optim is not None:
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    else:
        state = {'model': model.state_dict()}
    with open(final_model_path, 'wb') as f:
        torch.save(state, f)

    log_info = 'save final model.'
    logging(log_info, log_path)

except KeyboardInterrupt:
    log_info = '-' * 89
    logging(log_info, log_path)

    log_info = 'Exiting from training early'
    logging(log_info, log_path)

with open(model_path, 'rb') as f:
    state = torch.load(f)
model.load_state_dict(state['model'])

if args.model == 'LSTM':
    model.lstm.flatten_parameters()

test_loss = evaluate(test_data)

log_info = '=' * 89
logging(log_info, log_path)

log_info = '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss))
logging(log_info, log_path)

log_info = '=' * 89
logging(log_info, log_path)

with open(final_model_path, 'rb') as f:
    state = torch.load(f)
model.load_state_dict(state['model'])

test_loss = evaluate(test_data)

log_info = '=' * 89
logging(log_info, log_path)

log_info = '| Final model performance | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss))
logging(log_info, log_path)

log_info = '=' * 89
logging(log_info, log_path)