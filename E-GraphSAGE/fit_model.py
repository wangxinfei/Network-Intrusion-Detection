"""
@Author：Jinfan Zhang
@email：jzhan665@uottawa.ca
@Desc: main functions for training and testing
"""

import argparse
import numpy as np
import time
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from loader import (load_sage, load_gat)
from models import EGraphSage, PGD, ConsEGraphSage
import warnings
warnings.warn('ignore')

seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
data_class = {"UNSW-NB15":10,
              "Darknet":9,
              "CES-CIC":7,
              "ToN-IoT":10}
data_lr = {"UNSW-NB15":0.007,
           "Darknet":0.003,
           "CES-CIC":0.003,
           "ToN-IoT":0.01}
test_size = {"UNSW-NB15":210000,
             "Darknet":45000,
             "CES-CIC":75000,
             "ToN-IoT":140000}


def fit(args):
    alg = args.alg
    data = args.dataset
    binary = args.binary
    residual = args.residual
    path = "datasets/"+ data + '/'
    device = "cpu"  #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc2, edge_feat, label, node_map, adj = load_sage(path, binary)
    if args.cons or args.adv:
        model = ConsEGraphSage(data_class[data], enc2, edge_feat, node_map, adj, residual, args).to(device)
    else:
        model = EGraphSage(data_class[data], enc2, edge_feat, node_map, adj, residual, args).to(device)


    # loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            model.parameters()),
                                 lr=data_lr[data])


    if os.path.exists(path+'train.pkl'):
        train = torch.load(path+'train.pkl')
        val = torch.load(path + 'train.pkl')
        test = torch.load(path + 'train.pkl')

    else:  # save data, otherwise we would get different train data every time
        # train test split
        num_edges = len(edge_feat)
        train_val, test = train_test_split(np.arange(num_edges), test_size=test_size[data], stratify=label)
        train, val = train_test_split(train_val, test_size=5000, stratify=label[train_val])
        # print(len(train)) 316043
        random.shuffle(train)
        train = train[:5000]
        torch.save(train, path+'train.pkl')
        torch.save(val, path + 'train.pkl')
        torch.save(test, path + 'train.pkl')


    times = []
    trainscores = []
    valscores = []

    for epoch in range(3):
        print("Epoch: ", epoch)
        random.shuffle(train)
        epoch_start = time.time()
        for batch in range(int(len(train) / 500)):  # batches in train data
            batch_edges = train[500 * batch:500 * (batch + 1)]  # 500 records per batch
            start_time = time.time()
            # training
            model.train()
            output, _ = model(batch_edges)

            train_output = output.data.numpy()
            acc_train = f1_score(label[batch_edges],
                                 train_output.argmax(axis=1),
                                 average="weighted")
            loss = model.loss(batch_edges,
                              Variable(torch.LongTensor(label[np.array(batch_edges)])))

            optimizer.zero_grad()

            if args.adv:
                pgd = PGD(
                    eps=0.1,
                    alpha=0.02,
                    steps=20,
                    mode='targeted',
                    n_classes=10,
                )
                adv_inputs = pgd(inputs=batch_edges,
                                 labels=Variable(torch.LongTensor(label[np.array(batch_edges)])),
                                 pred=output,
                                 model=model)
                adv_otuput = model._forward(adv_inputs)
                adv_loss = model._loss(adv_otuput, Variable(torch.LongTensor(label[np.array(batch_edges)])))
                loss += adv_loss

            if args.cons:
                _, embeds = model(batch_edges.copy())
                kl_loss = F.kl_div(F.log_softmax(embeds, dim=-1), F.log_softmax(embeds, dim=-1)) + 0.01
                loss += kl_loss

            loss.backward()

            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            trainscores.append(acc_train)

            print('batch: {:03d}'.format(batch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'time: {:.4f}s'.format(end_time - start_time))

            if batch >= 10:
                break
        epoch_end = time.time()

        # Validation
        acc_val, loss_val, val_output = predict_(alg, model, label, loss_fn, val)
        valscores.append(acc_val)

        print('loss_val: {:.4f}'.format(loss_val),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'average batch time: {:.4f}s'.format(np.mean(times)),
              'epoch time: {:.2f}min'.format((epoch_end - epoch_start)/60.0))

    # Testing
    acc_test, loss_test, predict_output = predict_(alg, model, label, loss_fn, test)
    print("Test set results:", "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(acc_test.item()),
          "label acc=", f1_score(label[test], predict_output, average=None))


def predict_(alg, model, label, loss_fn, data_idx):
    predict_output = []
    loss = 0.0
    # emb = []
    for batch in range(int(len(data_idx) / 500)):
        batch_edges = data_idx[500 * batch:500 * (batch + 1)]

        batch_output,_ = model(batch_edges)

        batch_output = batch_output.data.numpy().argmax(axis=1)
        batch_loss = model.loss(batch_edges,
                                Variable(torch.LongTensor(label[np.array(batch_edges)])))

        predict_output.extend(batch_output)
        loss += batch_loss.item()
        # emb.append(embed)
    loss /= batch + 1
    acc = f1_score(label[data_idx], predict_output, average="weighted")
    # emb = torch.stack(emb).view(5000, -1)
    return acc, loss, predict_output


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    ALG = ['sage']
    DATA = ['UNSW-NB15', 'Darknet', 'CES-CIC', 'ToN-IoT']

    p = argparse.ArgumentParser()
    p.add_argument('--cons',
                   help='use_contrastive_learning',#False
                   default=True,
                   type=bool,)
    p.add_argument('--adv',
                   help='use_adversarial_perturb',
                   default=False,
                   type=bool,)
    p.add_argument('--focal',
                   help='use_focal_loss',
                   default=False,
                   type=bool,)
    p.add_argument('--alg',
                   help='algorithm to use.',
                   default='sage',
                   choices=ALG)
    p.add_argument('--dataset',
                   help='Experimental dataset.',
                   type=str,
                   default='ToN-IoT',
                   choices=DATA)
    p.add_argument('--binary',
                   help='Perform binary or muticlass task',
                   type=bool,
                   default=False)
    p.add_argument('--residual',
                   help='Apply modified model with residuals or not',
                   type=bool,
                   default=True)
    # Parse and validate script arguments.
    args = p.parse_args()

    # Training and testing
    fit(args)

# ToN-IoT
# ori 0.9230
# focal 0.9338
# adv 0.9686
# cons 0.8370
# adv+cons+focal 0.9683

# Darknet
# ori 0.8274
# focal 0.8302
# adv 0.8359
# cons 0.8370
# adv+cons+focal 0.8335


# CES
# ori 0.8023
# focal 0.8581
# adv 0.9407
# cons 0.9297
# adv+cons+focal 0.9440