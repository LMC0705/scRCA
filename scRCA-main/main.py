#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:42:55 2021

@author: liuyan
"""
# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from model import CNN
import argparse, sys
import numpy as np
import datetime
import shutil
from sklearn.metrics import f1_score
from numpy import *
from processing_celldata import read_cell,CellDataset
from loss import loss_coteaching

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.45)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default =0.35)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'ID, 10Xv3, or 10Xv2', default ='ID')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--pretrain_or_not', default=True)
parser.add_argument('--same_batch', default=False)
parser.add_argument('--different_batch', default=False)
parser.add_argument('--original_heatmap', default=False)
parser.add_argument('--draw_embedding', default=False)
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size =256
learning_rate = args.lr 

# load dataset


def precetrain(train_loader, model, optimizer, criterion=CE):

    model.train()

    for i, (input, target,idx) in enumerate(train_loader):
        if idx.size(0) != batch_size:
            continue

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def pre_train(model_ce,train_loader,epochs,learning_rate):
    for epoch in range(epochs):
    # if epoch >= 40:
    #     learning_rate = 0.001
    # if epoch >= 80:
    #     learning_rate = 0.0001
        optimizer_ce = torch.optim.SGD(model_ce.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)
        print("pretraning model_ce...", epoch)
        precetrain(train_loader=train_loader, model=model_ce, optimizer=optimizer_ce)


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def add_noise(inputs):
    # print (type(inputs))
    return inputs.float() + ((torch.randn(inputs.shape) *1)).float()
# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2):
    print ('Training %s...' % model_str)
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break
        # images=add_noise(images)
        images = images.cuda()
        labels = labels.cuda()
        
        # Forward + Backward + Optimize
        logits1=model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 2))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 2))
        train_total2+=1
        train_correct2+=prec2
        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
#        if (i+1) % args.print_freq == 0:
#            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f' 
#                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.data[0], loss_2.data[0], np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print ('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    pred_f=[]
    pred_g=[]
    true_label=[]
    for images, labels, _ in test_loader:
        images = images.cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()
        pred_f.extend(list(pred1.cpu().numpy()))
        true_label.extend(list(labels))
    #print (pred_f)
    print ("f net f1 score:",f1_score(true_label,pred_f, average='weighted'))    

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = images.cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
        pred_g.extend(list(pred2.cpu().numpy()))
    
    print ("g net f1 score:",f1_score(true_label,pred_g, average='weighted')) 
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2
def generate_single_batch_label(dataset_length,datasetname):
    batch_label=[]
    for i in range(dataset_length):
        batch_label.append(datasetname)
    return batch_label
def generate_all_batch_label(dataset_length,datasetname1):
    all_batch_label=[]
    for i in range(len(dataset_length)):
        single_batch=generate_single_batch_label(dataset_length[i],datasetname1[i])
        all_batch_label.extend(single_batch)
    return all_batch_label


def main():
    # Data Loader (Input Pipeline)
    if args.dataset=='10Xv2':
    platform="10Xv2"
    DataPath="paper_data/Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1.csv"
    LabelsPath="paper_data/Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1Labels.csv"
    CV_RDataPath="paper_data/Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1_folds.RData"
    if args.dataset=="10Xv3":
        platform="10Xv3"
        DataPath="paper_data/Inter-dataset/PbmcBench/10Xv3/10Xv3_pbmc1.csv"
        LabelsPath="paper_data/Inter-dataset/PbmcBench/10Xv3/10Xv3_pbmc1Labels.csv"
        CV_RDataPath="paper_data/Inter-dataset/PbmcBench/10Xv3/10Xv3_pbmc1_folds.RData"
    if args.dataset=="CL":
        platform="CL"
        DataPath="paper_data/Inter-dataset/PbmcBench/CEL-Seq/CL_pbmc1.csv"
        LabelsPath="paper_data/Inter-dataset/PbmcBench/CEL-Seq/CL_pbmc1Labels.csv"
        CV_RDataPath="paper_data/Inter-dataset/PbmcBench/CEL-Seq/CL_pbmc1_folds.RData"
    if args.dataset=="DR":
        platform="DR"
        DataPath="paper_data/Inter-dataset/PbmcBench/Drop-Seq/DR_pbmc1.csv"
        LabelsPath="paper_data/Inter-dataset/PbmcBench/Drop-Seq/DR_pbmc1Labels.csv"
        CV_RDataPath="paper_data/Inter-dataset/PbmcBench/Drop-Seq/DR_pbmc1_folds.RData"
if args.dataset=="ID":
    platform="ID"
    DataPath="paper_data/Inter-dataset/PbmcBench/inDrop/iD_pbmc1.csv"
    LabelsPath="paper_data/Inter-dataset/PbmcBench/inDrop/iD_pbmc1Labels.csv"
    CV_RDataPath="paper_data/Inter-dataset/PbmcBench/inDrop/iD_pbmc1_folds.RData"
#OutputDir="test"
train_x,train_label,y_train,test_x,test_label,y_test_cell_name,test_name,gene_names,cell_typeindex=read_cell(DataPath, LabelsPath, CV_RDataPath, platform)

num_classes=len(set(list(test_label)))


train_dataset = CellDataset(train_x, train_label,  
                                train=True, 
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                          )
    
test_dataset = CellDataset(test_x, test_label,  
                                train=False, 
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                          )

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not
print (noise_or_not)
# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
print (rate_schedule)
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
   
save_dir = args.result_dir +'/' +args.dataset+'/coteaching/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(args.noise_rate)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))

CE = nn.CrossEntropyLoss().cuda()
    print ('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print ('building model...')
    from CellNet import cellNet
    net_f = cellNet(train_x.shape[1],num_classes)

    # net_f.load_state_dict(torch.load("weights/10Xv2_45%.pt"))
    net_f.cuda()
    if args.pretrain_or_not:
        pre_train(net_f,train_loader,60,0.01)
    print (net_f.parameters)
    optimizer1 = torch.optim.Adam(net_f.parameters(), lr=learning_rate)
    
    net_g = cellNet(train_x.shape[1],num_classes)
    
    # net_g.load_state_dict(torch.load("weights/10Xv2_45%.pt"))
    net_g.cuda()
    if args.pretrain_or_not:
        pre_train(net_g,train_loader,30,0.01)
    print (net_g.parameters)
    optimizer2 = torch.optim.Adam(net_g.parameters(), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, net_f, net_g)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")

    # training
    model1=[]
    model2=[]
    for epoch in range(1, args.n_epoch):
        # train models
        net_f.train()
        adjust_learning_rate(optimizer1, epoch)
        net_g.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list=train(train_loader, epoch, net_f, optimizer1, net_g, optimizer2)
        # evaluate models
        test_acc1, test_acc2=evaluate(test_loader, net_f, net_g)
        # save results
        if epoch>15:
            model1.append(test_acc1)
            model2.append(test_acc2)
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")
    test_acc1, test_acc2=evaluate(test_loader, net_f, net_g)
    print (test_acc1,test_acc2)
    model_path="./weights/"+args.dataset+"_"+args.noise_type+"_"+str(args.noise_rate)+".pt"
    torch.save(net_g,model_path)
    if args.draw_embedding:
    
        from vision import draw_embedding,draw_batcheffectembedding
        net_g.eval()
        embeddings=net_g.forward_embedding(torch.tensor(test_x,dtype=torch.float32).cuda()).cpu().detach().numpy()
#    draw_embedding(embedding,list(y_test_cell_name["x"]),num_classes,args.dataset,args.noise_type,str(args.noise_rate))
        print ("the model 1 mean acc is",mean(model1))
        print ("the model 2 mean acc is",mean(model2))
        if args.dataset=="ID":
            length_data=[253,5927,2990,243,2991,3139,3067]
            datasetname=["SM2","10Xv2","10Xv3","CL","DR","SW","ID"]
            batch_label=generate_all_batch_label(length_data,datasetname)
        if args.dataset=="10Xv2":
            length_data=[253,6444,3222,253,3222,3176]
            datasetname=["SM2","10Xv2","10Xv3","CL","iD","SW"]
            batch_label=generate_all_batch_label(length_data,datasetname)
        if args.dataset=="10Xv3":
            length_data=[253,6406,253,3194,3212,3150]
            datasetname=["SM2","10Xv2","CL","DR","iD","SW"]
            batch_label=generate_all_batch_label(length_data,datasetname)
        from sklearn.manifold import TSNE
        tsne = TSNE()
    # X=tsne.fit_transform(embedding)
        embedding_tsne=tsne.fit_transform(embeddings)
    
        draw_embedding(embedding_tsne,list(y_test_cell_name["x"]),0,args.dataset,args.noise_type,str(args.noise_rate))
        draw_batcheffectembedding(embedding_tsne,batch_label,num_classes,args.dataset,args.noise_type,str(args.noise_rate))
    
    #############################draw bicluster#########################
    
    import random
    import plot_heatmap
    if args.same_batch:
        
        cell_index=random.sample(range(252,6000),100)
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_name[cell_index[i]])
            cell_labels.append(list(y_test_cell_name["x"])[cell_index[i]])
            embedding_cell_index=embeddings[cell_index,:]

        image_name="./heat_maps/"+args.dataset+args.noise_type+str(args.noise_rate)+"_samebatch_heat_map.eps"
        title="pbmc_"+args.dataset+args.noise_type+str(args.noise_rate)
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,image_name,title)
        
        if args.original_heatmap:
            original=test_x[cell_index,:]
            origin_name="./heat_maps/"+args.dataset+args.noise_type+str(args.noise_rate)+"_origina_samebatchlheat_map.eps"
            title="pbmc_"+args.dataset
            plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,original,origin_name,title)
        
    if args.different_batch:
        cell_index=random.sample(range(len(test_name)),100)
        cell_names=[]
        cell_labels=[]
        for i in range(len(cell_index)):
            cell_names.append(test_name[cell_index[i]])
            cell_labels.append(list(y_test_cell_name["x"])[cell_index[i]])
        embedding_cell_index=embeddings[cell_index,:]

        image_name="./heat_maps/"+args.dataset+args.noise_type+str(args.noise_rate)+"_heat_map.eps"
        title="pbmc_"+args.dataset+args.noise_type+str(args.noise_rate)
        plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,embedding_cell_index,image_name,title)
        if args.original_heatmap:
            original=test_x[cell_index,:]
            origin_name="./heat_maps/"+args.dataset+args.noise_type+str(args.noise_rate)+"_originalheat_map.eps"
            title="pbmc_"+args.dataset
            plot_heatmap.plot_cell_heatmap(cell_names,cell_labels,original,origin_name,title)

if __name__=='__main__':
    main()