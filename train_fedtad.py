import argparse
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from util.task_util import cal_topo_emb, accuracy, construct_graph, DiversityLoss
from util.base_util import seed_everything, load_dataset
from model import GCN, FedTAD_ConGenerator

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
    
    generator = FedTAD_ConGenerator(noise_dim=32, feat_dim=args.hid_dim if args.fedtad_mode == 'rep_distill' else subgraphs[0].x.shape[1], out_dim=dataset.num_classes, dropout=0).to(device)
    global_optimizer = Adam(global_model.parameters(), lr=args.lr_d, weight_decay=args.weight_decay)
    generator_optimizer = Adam(generator.parameters(), lr=args.lr_g, weight_decay=args.weight_decay)

    best_server_val = 0
    best_server_test = 0


    if os.path.exists(f"./ckr/{args.dataset}_{args.partition}_{args.num_clients}.pt"):
        ckr = torch.load(f"./ckr/{args.dataset}_{args.partition}_{args.num_clients}.pt").to(device)
    else:
        os.makedirs("./ckr", exist_ok=True)
        ckr = torch.zeros((args.num_clients, dataset.num_classes)).to(device)
        for client_id in range(args.num_clients):
            data = subgraphs[client_id]  
            graph_emb = cal_topo_emb(edge_index=data.edge_index, num_nodes=data.x.shape[0], max_walk_length=5).to(device)    
            ft_emb = torch.cat((data.x, graph_emb), dim=1).to(device)
            for train_i in data.train_idx.nonzero().squeeze():
                neighbor = data.edge_index[1,:][data.edge_index[0, :] == train_i] 
                node_all = 0
                for neighbor_j in neighbor:
                    node_kr = torch.cosine_similarity(ft_emb[train_i], ft_emb[neighbor_j], dim=0)
                    node_all += node_kr
                node_all += 1
                node_all /= (neighbor.shape[0] + 1)
                
                label = data.y[train_i]
                ckr[client_id, label] += node_all
        torch.save(ckr, f"./ckr/{args.dataset}_{args.partition}_{args.num_clients}.pt")
    
    
    normalized_ckr = ckr / ckr.sum(0)
    
    
    l_glb_acc_test = []
    
    for round_id in range(args.num_rounds):
        global_model.eval()
        generator.eval()
        
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
        
        
        l_glb_acc_test.append(global_acc_test)
        
        
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
        
        
        num_gen = args.num_gen
        c_cnt = [0] * dataset.num_classes
        for class_i in range(dataset.num_classes):
            c_cnt[class_i] = int(num_gen * 1 / dataset.num_classes)
        c_cnt[-1] += num_gen - sum(c_cnt)

        print(f"pseudo label distribution: {c_cnt}")
        c = torch.zeros(num_gen).to(device).long()
        ptr = 0
        for class_i in range(dataset.num_classes):
            for _ in range(c_cnt[class_i]):
                c[ptr] = class_i
                ptr += 1
                
                
        each_class_idx = {}
        for class_i in range(dataset.num_classes):
            each_class_idx[class_i] = c == class_i
            each_class_idx[class_i] = each_class_idx[class_i].to(device)


        
        for client_id in range(args.num_clients):
            local_models[client_id].eval()
        
        
        for _ in range(args.glb_epochs):
            
            ############ sampling noise ##############
            z = torch.randn((num_gen, 32)).to(device)
            
            
            ############ train generator ##############
            generator.train()
            global_model.eval()
            for it_g in range(args.it_g):
                loss_sem = 0
                loss_diverg = 0
                loss_div = 0
                
                
                generator_optimizer.zero_grad()
                for client_id in range(args.num_clients):
                    ######  generator forward  ########
                    node_logits = generator.forward(z=z, c=c) 
                    node_norm = F.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(
                        node_logits, adj_logits, k=args.topk)
                    
                    ##### local & global model -> forward #########
                    if args.fedtad_mode == 'rep_distill':
                        local_pred = local_models[client_id].rep_forward(
                            data=pseudo_graph)
                        global_pred = global_model.rep_forward(
                            data=pseudo_graph)
                    else:
                        local_pred = local_models[client_id].forward(
                            data=pseudo_graph)
                        global_pred = global_model.forward(
                            data=pseudo_graph)  
                        
                        
                    ##########  semantic loss  #############
                    acc_list = [0] * dataset.num_classes
                    for class_i in range(dataset.num_classes):
                        loss_sem += normalized_ckr[client_id][class_i] * nn.CrossEntropyLoss()(local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
                        acc = accuracy(local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
                        acc_list[class_i] = f"{acc:.2f}"
                    acc_tot = float(accuracy(local_pred, c))
                    # print(f"[client {client_id}] accuracy on each class for pseudo_graph: {acc_list}, on all classes: {acc_tot:.2f}")


                    ############  diversity loss  ##############
                    loss_div += DiversityLoss(metric='l1').to(device)(z.view(z.shape[0],-1), node_logits) 
                
                
                    ############  divergence loss  ############   
                    for class_i in range(dataset.num_classes):
                        loss_diverg += - normalized_ckr[client_id][class_i] * torch.mean(torch.mean(
                            torch.abs(global_pred[each_class_idx[class_i]] - local_pred[each_class_idx[class_i]].detach()), dim=1))


                ############ generator loss #############
                loss_G = args.lam1 * loss_sem + loss_diverg + args.lam2 * loss_div              
                # print(f'[generator] loss_sem: {loss_sem:.4f}\tloss_div: {loss_div:.4f}\tloss_diverg: {loss_diverg:.4f}\tloss_G: {loss_G:.4f}')
                
                loss_G.backward()
                generator_optimizer.step()
                    
                    
                    
                
            ########### train global model ###########

            
            generator.eval()
            global_model.train()
            
            
            ######  generator forward  ########
                    node_logits = generator.forward(z=z, c=c)                    
                    node_norm = F.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(node_logits.detach(), adj_logits.detach(), k=args.topk)
            
            for it_d in range(args.it_d):
                global_optimizer.zero_grad()
                loss_D = 0
                
                for client_id in range(args.num_clients):    
                    #######  local & global model -> forward  #######
                    if args.fedtad_mode == 'rep_distill':
                        local_pred = local_models[client_id].rep_forward(
                            data=pseudo_graph)
                        global_pred = global_model.rep_forward(
                            data=pseudo_graph)
                    else:
                        local_pred = local_models[client_id].forward(
                            data=pseudo_graph)
                        global_pred = global_model.forward(
                            data=pseudo_graph)  
                    
                    ############  divergence loss  ############   
                    
                    for class_i in range(dataset.num_classes):
                        loss_D += normalized_ckr[client_id][class_i] * torch.mean(torch.mean(
                            torch.abs(global_pred[each_class_idx[class_i]] - local_pred[each_class_idx[class_i]].detach()), dim=1))

                
                
                # print(f"[global] loss_diverg: {loss_D:.4f}")
                loss_D.backward()
                global_optimizer.step()

