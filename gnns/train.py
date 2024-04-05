import sys
from os.path import dirname
sys.path.append(dirname(__file__))
sys.path.append(".")

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from seo_import import *
from sklearn.metrics import  accuracy_score, f1_score
from torch_geometric.seed import seed_everything
from sklearn.metrics import mean_squared_error, r2_score
#from torch.utils.tensorboard import SummaryWriter
from model import GNN_v2, GNN_v1, LinModel
# from gnns.node2vec import *
import random
from datetime import datetime

EPOCHS = 10000
# for balanced, lets make it a bit random
seed_everything(42)

# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_params():
    with torch.no_grad():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(device)
        model(batch.x, batch.edge_index, batch.edge_weight)

# loss_fn = torch.nn.L1Loss()
loss_fn = torch.nn.MSELoss()

def train(train_loader, epoch):
    """
    Trains the model for a regression task.
    """
    model.train()
    total_loss = 0
    predictions = []
    true_values = []
    train_loader_len = 0
    
    for batch in train_loader:
        batch_size = batch.batch_size
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_weight)[:batch_size]
        
        #loss_fn(out.squeeze(), batch.y[:batch_size])  # Ensure correct shape and type for y
        
        loss = torch.mean((out.squeeze() - batch.y[:batch_size])**2)
        
        
        #l1_norm = sum(p.abs().sum() for p in model.parameters())
        # loss += 1e6 * l1_norm

        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        train_loader_len += 1
        
        predictions.extend(out.detach().cpu().numpy())
        true_values.extend(batch.y[:batch_size].cpu().numpy())
#     # Perform a single optimization step after processing all batches
#     optimizer.step()
#     optimizer.zero_grad()

    train_loss = total_loss / train_loader_len
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    #writer.add_scalar('Loss/Train', train_loss, epoch)
    #writer.add_scalar('MSE/Train', mse, epoch)
    #writer.add_scalar('R2/Train', r2, epoch)

    return train_loss, mse, r2


def test(loader, epoch):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        train_loader_len = 0
        predictions = []
        true_values = []
        
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.batch_size
            out = model(batch.x, batch.edge_index, batch.edge_weight)[:batch_size]
            
            # No need for argmax in regression
            total_loss += torch.mean((out.squeeze() - batch.y[:batch_size])**2).detach().cpu()

            train_loader_len += 1
            
            predictions.extend(out.detach().cpu().numpy())
            true_values.extend(batch.y[:batch_size].cpu().numpy())

    valid_loss = total_loss / train_loader_len
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    #writer.add_scalar('Loss/Validation', valid_loss, epoch)
    #writer.add_scalar('MSE/Validation', mse, epoch)
    #writer.add_scalar('R2/Validation', r2, epoch)

    return valid_loss, mse, r2


class EarlyStopping():
    def __init__(self, patience=10, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.best_epoch = 0

    def __call__(self, train_loss, validation_loss):
        
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = 0
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:# and self.counter > 40:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



if __name__ == '__main__':
    ## Run the script
    training_stats = {}
    final_pds = []
    time = str(datetime.now())
    #writer = SummaryWriter(f'runs/{task}/{network}_{weight_scheme}_{time}')
    outdfs = []
    training_stats['log_links'] = []
    ## HOMOGENOUS LINKS ONLY RUNS:
    # we can use these loaders when we run the first epoch
    data_input_path = '../data/3kupdate/attributes_3k.csv'
    links_input_path = '../data/3kupdate/filtered_backlinks.csv'
    #task = 'cc-orig-pr'
    task = 'traffic'
    labels_input_path = '../data/traffic.csv'
    weight_scheme = 'log_links'                
    labelled, links, url_mapper = import_seo_links(data_input_path, links_input_path, task, labels_input_path)

    train_loader, valid_loader, test_loader = train_val_test_split(labelled, links, weight_scheme, task, batch_size=64)

    # for node2vec (only) homogenous runs
    #ulabs, el, url_mapper = import_seo_and_users(edge_path, attr_path)
    #data = to_pt_n2v_tensors(ulabs, el)
    #train_loader, valid_loader, test_loader= node_masker_n2v_hetweights(edge_path, attr_path, embedding_path)
    for _ in range(10):
        final_pds = []
        for run in ['linear', 'gnn_uw', 'gnn_w']:
            if run == 'linear':
                model = LinModel(23, 1)#, use_weights=False)
            elif run == 'gnn_uw':
                model = GNN_v1(23, 1, use_weights = False)
            elif run == 'gnn_w':
                model = GNN_v1(23,1, use_weights=True)
            model = model.to(device)
            init_params()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
            loss_fn = torch.nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, 2e-5)
            # early_stopping = EarlyStopping(patience=15, min_delta=0.0001)
            early_stopping = EarlyStopping(patience=100, min_delta=0.0001)
            for epoch in tqdm(range(0, EPOCHS)):

                tloss, tmse, tr2 = train(train_loader, epoch)
                #print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
                vloss, vmse, vr2 = test(valid_loader, epoch)
                test_loss, test_mse, test_r2 = test(test_loader, epoch)
                scheduler.step()
                print(f'Epoch: {epoch}')
                print(f'TrainMSE: {tmse:.4f}, Val: {vmse:.4f}, T_loss: {tloss:.4f} V_loss: {vloss:.4f}')

                # Record all statistics from this epoch.
                training_stats[weight_scheme].append(
                    {
                        'epoch': epoch + 1,
                        'Training MSE': tmse,
                        'Training Loss': tloss,
                        'Training R2': tr2,
                        'Valid. Loss': vloss,
                        'Valid. MSE.': vmse,
                        'Valid. R2': vr2,
                        'Test MSE' : test_mse,
                        'Test R2': test_r2,
                        'Test Loss': test_loss
                    }
                )
                
                early_stopping(tloss, test_mse)
                if early_stopping.early_stop:
                    print("We are at epoch:", epoch)
                    patience = early_stopping.patience
                    acc_csv = {
                        'task': task,
                        'weight_scheme': weight_scheme,
                        # 'validation_mse' : training_stats[weight_scheme][epoch-patience]['Valid. MSE.'],
                        # 'val_r2':training_stats[weight_scheme][epoch-patience]['Valid. R2'],
                        'training_mse': training_stats[weight_scheme][epoch-patience]['Training MSE'],
                        'train_r2': training_stats[weight_scheme][epoch-patience]['Training R2'], 
                        'test_mse': training_stats[weight_scheme][epoch-patience]['Test MSE'],
                        'test_r2':training_stats[weight_scheme][epoch-patience]['Test R2'],
                        'epoch': epoch - patience,
                        'run': run,
                    }
                    break
            
            final_pds.append(acc_csv)
        outdfs.append(pd.DataFrame(final_pds))
        #pd.DataFrame(final_pds).to_csv(output_prefix + '_results.csv')
        #pd.DataFrame(final_pds).to_csv(output_prefix + '_results.csv', index=False)

hm = pd.concat(outdfs)
hm.to_csv('log_tr_results.csv', index = False)
condensed = hm.groupby(['run']).agg({'test_mse':['mean', 'std']}).reset_index()
condensed.to_csv('logtr_results_condensed.csv', index = False)