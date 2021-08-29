from __future__ import division
from __future__ import print_function
import os
import random
import argparse
import numpy as np
import time
import torch
import math
from sklearn.utils import shuffle
from utils import load_data,get_metrics
from models import NodeP
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=36, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=150, help='Number of hidden units.')
parser.add_argument('--b_sz', type=int, default=70, help='Number of batch_size.')
parser.add_argument('--class_num', type=int, default=4, help='Number of labels')
args = parser.parse_args()
torch.cuda.set_device(1)
test_single=True
rng = np.random.RandomState(seed=args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
features, labels,label_emb,neighbor = load_data()

model = NodeP(batch_size=args.b_sz,
              lstm_hid_dim= args.hidden,
              n_classes=args.class_num,
              label_embed=label_emb,
              )

model=model.cuda()
features=features.cuda()

def train(model,idx_train, idx_val,crit,opt,epochs,b_sz,neighbor,features):
    # if not os.path.isdir('./sala_log'):
    #     os.makedirs('./sala_log')
    # trace_file = './sala_log/trace.txt'
    for epoch in range(epochs):
        print("Running EPOCH",epoch+1)
        train_loss = []
        micro_f1_ep = []
        micro_precision_ep = []
        micro_recall_ep = []
        train_nodes = shuffle(idx_train)
        batches = math.floor(len(train_nodes) / b_sz)
        for index in range(batches):
            opt.zero_grad()
            trn_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            labels_batch = torch.FloatTensor(labels[trn_batch]).cuda()
            trn_node_emb=features[trn_batch]
            trn_batch_neighbor = neighbor[trn_batch].numpy()
            #y_pred,weight_all,weight_one= attention_model(x)
            trn_neighbor_emb=features[trn_batch_neighbor]
            y_pred= model(trn_node_emb,trn_neighbor_emb)

            loss = crit(y_pred, labels_batch.float())
            loss.backward()
            opt.step()
            labels_cpu = labels_batch.data.cpu().float()
            pred_cpu = y_pred.data.cpu()
            pred_2 = np.int64(pred_cpu.numpy() > 0.5)
            hamming_loss, micro_f1, micro_precision, micro_recall = get_metrics(labels_cpu.numpy(), pred_2)
            train_loss.append(float(loss))
            micro_f1_ep.append(micro_f1)
            micro_precision_ep.append(micro_precision)
            micro_recall_ep.append(micro_recall)
        avg_loss = np.mean(train_loss)
        avg_micro_f1 = np.mean(micro_f1_ep)
        avg_micro_precision = np.mean(micro_precision_ep)
        avg_micro_recall = np.mean(micro_recall_ep)
        print("epoch %2d train end : avg_loss = %.4f" % (epoch+1, avg_loss))
        print(' micro_f1: %.4f | micro_precision: %.4f | micro_recall: %.4f'
              % ( avg_micro_f1, avg_micro_precision, avg_micro_recall))
        print("start validation!!!!!")
        test_loss = []
        test_hamming = []
        test_micro_prec = []
        test_micro_recall = []
        test_micro_f1 = []
        val_nodes = shuffle(idx_val)
        batches = math.floor(len(val_nodes) / b_sz)
        for index in range(batches):
            t = time.time()
            test_batch = val_nodes[index * b_sz:(index + 1) * b_sz]

            test_labels_batch = torch.FloatTensor(labels[test_batch]).cuda()

            tst_node_embs = features[test_batch]

            tst_batch_neighbor = neighbor[test_batch].numpy()
            tst_neighbor_emb = features[tst_batch_neighbor]
            val_y= model(tst_node_embs, tst_neighbor_emb)

            loss = crit(val_y, test_labels_batch.float())/b_sz
            labels_cpu = test_labels_batch.data.cpu().float()

            pred_cpu = val_y.data.cpu()
            pred_2 =  np.int64(pred_cpu.numpy() > 0.5)
            hamming_loss, micro_f1, micro_precision, micro_recall = get_metrics(labels_cpu.numpy(), pred_2)
            test_loss.append(float(loss))
            test_hamming.append(hamming_loss)
            test_micro_prec.append(micro_precision)
            test_micro_recall.append(micro_recall)
            test_micro_f1.append(micro_f1)
        avg_test_loss = np.mean(test_loss)
        avg_test_micro_prec = np.mean(test_micro_prec)
        avg_test_micro_recall = np.mean(test_micro_recall)
        avg_test_micro_f1 = np.mean(test_micro_f1)
        print("epoch %2d test end : avg_loss = %.4f" % (epoch+1, avg_test_loss))
        print('micro_f1: %.4f | micro_precision: %.4f | micro_recall: %.4f'
              % ( avg_test_micro_f1, avg_test_micro_prec, avg_test_micro_recall))
        print('time: {:.4f}s'.format(time.time() - t))
        # if epoch % 5 == 0:
        #     p = './sala_log/best_%d.pth' % epoch
        #     name = model.save(path=p)
        #     print("save done", name)
def iterative_sampling(Y, labeled_idx, fold, rng):
    ratio_per_fold = 1 / fold
    folds = [[] for i in range(fold)]
    number_of_examples_per_fold = np.array([(1 / fold) * np.shape(Y[labeled_idx, :])[0] for i in range(fold)])

    blacklist_samples = np.array([])
    number_of_examples_per_label = np.sum(Y[labeled_idx, :], 0)
    blacklist_labels = np.where(number_of_examples_per_label < fold)[0]
    print(blacklist_labels)
    desired_examples_per_label = number_of_examples_per_label * ratio_per_fold

    subset_label_desire = np.array([desired_examples_per_label for i in range(fold)])
    total_index = np.sum(labeled_idx)
    max_label_occurance = np.max(number_of_examples_per_label) + 1
    sel_labels = np.setdiff1d(range(Y.shape[1]), blacklist_labels)

    while total_index > 0:
        try:
            min_label_index = np.where(number_of_examples_per_label == np.min(number_of_examples_per_label))[0]
            for index in labeled_idx:
                if (Y[index, min_label_index[0]] == 1 and index != -1) and (min_label_index[0] not in blacklist_labels):
                    m = np.where(
                        subset_label_desire[:, min_label_index[0]] == subset_label_desire[:, min_label_index[0]].max())[0]
                    if len(m) == 1:
                        folds[m[0]].append(index)
                        subset_label_desire[m[0], Y[index, :].astype(np.bool)] -= 1
                        labeled_idx[np.where(labeled_idx == index)] = -1
                        number_of_examples_per_fold[m[0]] -= 1
                        total_index = total_index - index
                    else:
                        m2 = np.where(number_of_examples_per_fold[m] == np.max(number_of_examples_per_fold[m]))[0]
                        if len(m2) > 1:
                            m = m[rng.choice(m2, 1)[0]]
                            folds[m].append(index)
                            subset_label_desire[m, Y[index, :].astype(np.bool)] -= 1
                            labeled_idx[np.where(labeled_idx == index)] = -1
                            number_of_examples_per_fold[m] -= 1
                            total_index = total_index - index
                        else:
                            m = m[m2[0]]
                            folds[m].append(index)
                            subset_label_desire[m, Y[index, :].astype(np.bool)] -= 1
                            labeled_idx[np.where(labeled_idx == index)] = -1
                            number_of_examples_per_fold[m] -= 1
                            total_index = total_index - index
                elif (Y[index, min_label_index[0]] == 1 and index != -1):
                    if (min_label_index[0] in blacklist_labels) and np.any(Y[index, sel_labels]) == False:
                        np.append(blacklist_samples, index)
      
                        labeled_idx[np.where(labeled_idx == index)] = -1 
                        total_index = total_index - index

            number_of_examples_per_label[min_label_index[0]] = max_label_occurance
        except:
            traceback.print_exc(file=sys.stdout)
            exit()

    Y = Y[:, sel_labels]

    return folds, Y, blacklist_samples
crit = torch.nn.BCELoss())
opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
labeled_idx = np.where(labels.sum(-1) > 0)[0]
num_runs = 5
cv_splits, Y, blacklist_samples = iterative_sampling(labels, labeled_idx, num_runs, rng)
print("Done loading training data..")
for i in range(num_runs):
    training_samples = []
    testing_samples = []
    for j in range(len(cv_splits)):
        if test_single:
            if j != i:
                training_samples += cv_splits[j]
            else:
                testing_samples = cv_splits[j]
        else:
            if j == i:
                training_samples = cv_splits[j]
            else:
                testing_samples += cv_splits[j]
    train(model,training_samples,testing_samples,crit,opt,args.epochs,args.b_sz,neighbor,features)
