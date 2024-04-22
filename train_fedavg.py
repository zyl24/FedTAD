import argparse
import warnings
import torch
import torch.nn as nn
from torch.optim import Adam
from util.task_util import accuracy
from util.base_util import seed_everything, load_dataset
from model import GCN

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()


# experimental environment setup
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--root', type=str, default='/home/ai2/work/fedtad/dataset')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--dataset', type=str, default="Cora")
parser.add_argument('--partition', type=str, default="Louvain", choices=["Louvain", "Metis"])
parser.add_argument('--part_delta', type=int, default=20)
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--num_rounds', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--num_dims', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hid_dim', type=int, default=64)



# for fedtad
parser.add_argument('--glb_epochs', type=int, default=5)
parser.add_argument('--it_g', type=int, default=1)
parser.add_argument('--it_d', type=int, default=5)
parser.add_argument('--lr_g', type=float, default=1e-3)
parser.add_argument('--lr_d', type=float, default=1e-3)
parser.add_argument('--fedtad_mode', type=str, default='raw_distill', choices=['raw_distill', 'rep_distill'])
parser.add_argument('--num_gen', type=int, default=100)
parser.add_argument('--lam1', type=float, default=1)
parser.add_argument('--lam2', type=float, default=1)
parser.add_argument('--topk', type=float, default=5)




args = parser.parse_args()






    

    
    
    
    
    

if __name__ == "__main__":
    
    seed_everything(seed=args.seed)
    dataset = load_dataset(args)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device(f"cuda:{args.gpu_id}")
    subgraphs = [dataset.subgraphs[client_id].to(device) for client_id in range(args.num_clients)]
    local_models = [ GCN(feat_dim=subgraphs[client_id].x.shape[1], 
                         hid_dim=args.hid_dim, 
                         out_dim=dataset.num_classes,
                         dropout=args.dropout).to(device)
                    for client_id in range(args.num_clients)]
    local_optimizers = [Adam(local_models[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay) for client_id in range(args.num_clients)]
    global_model = GCN(feat_dim=subgraphs[0].x.shape[1], 
                         hid_dim=args.hid_dim, 
                         out_dim=dataset.num_classes,
                         dropout=args.dropout).to(device)

    best_server_val = 0
    best_server_test = 0


    
    for round_id in range(args.num_rounds):
        # global model broadcast
        for client_id in range(args.num_clients):
            local_models[client_id].load_state_dict(global_model.state_dict())
        
        
        # global eval
        global_acc_val = 0
        global_acc_test = 0
        for client_id in range(args.num_clients):

            local_models[client_id].eval()
            logits = local_models[client_id].forward(subgraphs[client_id])
            loss_train = loss_fn(logits[subgraphs[client_id].train_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].train_idx])
            loss_val = loss_fn(logits[subgraphs[client_id].val_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].val_idx])
            loss_test = loss_fn(logits[subgraphs[client_id].test_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].test_idx])
            acc_train = accuracy(logits[subgraphs[client_id].train_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].train_idx])
            acc_val = accuracy(logits[subgraphs[client_id].val_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].val_idx])
            acc_test = accuracy(logits[subgraphs[client_id].test_idx], 
                            subgraphs[client_id].y[subgraphs[client_id].test_idx])
            
            print(f"[client {client_id}]: acc_train: {acc_train:.2f}\tacc_val: {acc_val:.2f}\tacc_test: {acc_test:.2f}\tloss_train: {loss_train:.4f}\tloss_val: {loss_val:.4f}\tloss_test: {loss_test:.4f}")
            global_acc_val += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_val
            global_acc_test += subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] * acc_test
            
        print(f"[server]: current_round: {round_id}\tglobal_val: {global_acc_val:.2f}\tglobal_test: {global_acc_test:.2f}")
        
        if global_acc_val > best_server_val:
            best_server_val = global_acc_val
            best_server_test = global_acc_test
            best_round = round_id
        print(f"[server]: best_round: {best_round}\tbest_val: {best_server_val:.2f}\tbest_test: {best_server_test:.2f}")
        print("-"*50)
        
        
        
        
        # local train
        for client_id in range(args.num_clients):
            for epoch_id in range(args.num_epochs):
                local_models[client_id].train()
                local_optimizers[client_id].zero_grad()
                
                logits = local_models[client_id].forward(subgraphs[client_id])
                loss_train = loss_fn(logits[subgraphs[client_id].train_idx], 
                               subgraphs[client_id].y[subgraphs[client_id].train_idx])
                loss_train.backward()
                local_optimizers[client_id].step()
                
        # global aggregation
        with torch.no_grad():
            for client_id in range(args.num_clients):
                weight = subgraphs[client_id].x.shape[0] / dataset.global_data.x.shape[0] 
                for (local_state, global_state) in zip(local_models[client_id].parameters(), global_model.parameters()):
                    if client_id == 0:
                        global_state.data = weight * local_state
                    else:
                        global_state.data += weight * local_state
        
      