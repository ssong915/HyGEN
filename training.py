import torch
from sklearn import metrics
from utils import *
import torch.nn.functional as F

def model_train(args, g, model, Aggregator, Generator, optim_D, optim_G, pos_hedges, pos_labels, train_pred, train_label, device, epoch):
    batch_size = len(pos_hedges)
    
    if epoch % args.train_DG[2] < args.train_DG[0]: # update D 
        trainD, trainG = True, False
    else: # update G
        trainD, trainG = False, True      

    if trainD:
        model.train()
        Aggregator.train()
        optim_D.zero_grad()
    else:
        model.eval()
        Aggregator.eval()
        
    # message passing
    if args.model == "hnhn":
        v_feat = args.v_feat[g.nodes('node')]
        e_feat = args.e_feat[g.nodes('edge')]
        v_reg_weight = args.v_reg_weight[g.nodes('node')]
        v_reg_sum = args.v_reg_sum[g.nodes('node')]
        e_reg_weight = args.e_reg_weight[g.nodes('edge')]
        e_reg_sum = args.e_reg_sum[g.nodes('edge')]
        v, e = model([g, v_feat, e_feat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum], args.n_layers)

    elif args.model == 'hgnn':
        v_feat = args.v_feat[g.nodes('node')]
        incidence = args.HGNN_G
        matrix_np = np.array(incidence)
        incidence = matrix_np.tolist()
        incidence = torch.tensor(incidence)
        incidence = incidence.to(device) 
        v = model(v_feat, incidence)

    if trainG:
        Generator.train()
        optim_G.zero_grad()
    else:
        Generator.eval()
        
    ## Train Generator ##
    pos_preds = []
    pos_embed = []
    for hedge in pos_hedges :
        embeddings = v[hedge]
        pred, embed = Aggregator(embeddings)
        pos_preds.append(pred)
        pos_embed.append(embed.cpu())
        train_pred.append(pred.detach())
    train_label.append(pos_labels.detach())
    pos_labels = pos_labels.type(torch.FloatTensor).to(device)
    pos_preds = torch.stack(pos_preds)
    pos_preds = pos_preds.squeeze()

    np_p_hedge_embeddings = np.array([tensor.detach().numpy() for tensor in pos_embed])
    p_hedge_embeddings = torch.from_numpy(np_p_hedge_embeddings).to(device)

    pos_hedge_onehots, neg_hedge_onehots, neg_hedge_indices = Generator(args, batch_size, pos_hedges, p_hedge_embeddings)
            
    neg_onehot = []
    neg_preds = []
    neg_embed = []

    for i, onehots in enumerate(neg_hedge_onehots) :
        count = torch.sum(onehots, dim=0)
        inv_freq = torch.where(count>0, 1/count, torch.zeros_like(count)).detach()
        onehots = onehots*inv_freq.repeat(onehots.shape[0], 1)
        embeddings = torch.matmul(onehots, v)
        pred, embed = Aggregator(embeddings)
        neg_preds.append(pred)
        neg_embed.append(embed)
        train_pred.append(pred.detach())
        neg_onehot.append(torch.sum(onehots, dim=0).detach())
    
    neg_labels = torch.zeros(len(neg_onehot)).to(device)
        
    if args.memory_size > 0:
        memory_embed = []
        for i in range(model.memory_size) :
            onehot = model.memory[i]
            onehots = unsqueeze_onehot(onehot)
            embeddings = torch.matmul(onehots, v)
            pred, embed = Aggregator(embeddings)
            neg_preds.append(pred)
            neg_embed.append(embed)
            memory_embed.append(embed)
        memory_candidate = torch.cat((torch.stack(neg_onehot), model.memory), dim=0)
        memory_embed = torch.stack(memory_embed)
        memory_preds = torch.stack(neg_preds).detach().squeeze()

    train_label.append(neg_labels.detach())
    neg_preds = torch.stack(neg_preds)
    
    neg_preds = neg_preds.squeeze().to(device)
    
    d_real_loss = -torch.mean(pos_preds)
    d_fake_loss = torch.mean(neg_preds)
    g_loss = -torch.mean(neg_preds)   
    d_loss = (d_real_loss + d_fake_loss) /2 
    
    if args.loss_used=='used':
        neg_embed = torch.stack(neg_embed[:len(pos_embed)])
        pos_embed = torch.stack(pos_embed)
        reg_loss = nt_xent(args, pos_embed.cpu(), neg_embed.cpu())
        d_loss = d_loss + reg_loss
        
    if trainD:
        d_loss.backward()
        optim_D.step()
    
    if trainG:
        g_loss.backward()
        optim_G.step()
        
    if args.training == "wgan":
        for _, param in model.named_parameters():
            param.clamp(-args.clip,args.clip)
        for _, param in Aggregator.named_parameters():
            param.clamp(-args.clip,args.clip)
      
    if args.memory_size > 0:
        if args.memory_type == "sample":
            indices = memory_preds.topk(k=memory_preds.shape[0]).indices
            model.memory = memory_candidate[indices[:args.memory_size]]
            replaced_indice = indices[:args.memory_size]
            replaced_indice[replaced_indice>=args.bs]=0

    np_n_hedge_embeddings = [tensor.detach().cpu() for tensor in neg_embed]
    np_n_hedge_embeddings = np.array([tensor.numpy() for tensor in np_n_hedge_embeddings])
    
    return d_loss.item(), g_loss.item(), train_pred, train_label, pos_hedge_onehots, neg_onehot
            
def model_eval(args, test_batchloader, g, model, Aggregator,device):
    model.eval()
    Aggregator.eval()
    with torch.no_grad():
        total_pred = []
        test_label = []
        total_embedddings = []
        test_hedges = []
        num_data = 0

        while True :
            hedges, labels, is_last = test_batchloader.next()
            batch_size = len(hedges)
            num_data+=batch_size

            if args.model == "hnhn":
                v_feat = args.v_feat[g.nodes('node')]
                e_feat = args.e_feat[g.nodes('edge')]
                v_reg_weight = args.v_reg_weight[g.nodes('node')]
                v_reg_sum = args.v_reg_sum[g.nodes('node')]
                e_reg_weight = args.e_reg_weight[g.nodes('edge')]
                e_reg_sum = args.e_reg_sum[g.nodes('edge')]
                v, e = model([g, v_feat, e_feat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum], args.n_layers)
            
            elif args.model == 'hgnn':
                v_feat = args.v_feat[g.nodes('node')]
                incidence = args.HGNN_G
                matrix_np = np.array(incidence)
                incidence = matrix_np.tolist()
                incidence = torch.tensor(incidence)
                incidence = incidence.to(device) 
                v = model(v_feat, incidence)
            else:
                assert()
                
            for hedge in hedges :
                test_hedges.append(hedge)
                embeddings = v[hedge]
                pred, test_embed = Aggregator(embeddings) 
                total_embedddings.append(test_embed)
                total_pred.append(pred.detach())
            test_label.append(labels.detach())
            
            if is_last :
                break
        total_pred = torch.stack(total_pred)       
        total_pred = torch.sigmoid(total_pred.squeeze())
        test_label = torch.cat(test_label, dim=0)
        
        test_hedge_embeddings = [tensor.detach().cpu() for tensor in total_embedddings]
        test_hedge_embeddings = np.array([tensor.numpy() for tensor in test_hedge_embeddings])
        

    return total_pred.tolist(), test_label.tolist(), test_hedge_embeddings, test_hedges
