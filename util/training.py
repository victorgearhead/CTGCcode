from util.utils import *
from util.module import *
from util.models import *



def pre_train(args, data, data_test, model, load=True):
    # pretrain by Node Discriminate
    save_path = args.model_path + args.dataset_name + "_pre.pt"
    if os.path.exists(save_path):
        param = torch.load(save_path)
        model_dict = model.state_dict()
        filtered_param = {k: v for k,v in param.items() if 'classifier' not in k}
        model_dict.update(filtered_param)
        model.load_state_dict(model_dict)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        n = data.x.shape[0]
        disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0).to(args.device)

        best_loss = 1e6
        nmi=0.
        for epoch in range(1, args.epoch_pretrain):
            model.train()
            output = model.SSL_dis(data)
            loss = criterion(output, disc_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < best_loss:
                best_loss = loss
                weight = model.state_dict()
                with torch.no_grad():
                    model.eval()
                    # Skip clustering evaluation for multi-label datasets (e.g., PPI)
                    if args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1):
                        nmi, ari = 0., 0.
                    else:
                        if args.dataset_name in ['flickr', 'reddit']:
                            H = model.embedding(data_test)
                            labels_test = data_test.y
                            cls_num = int(labels_test.max()+1)
                        else:
                            H = model.embedding(data)[data.test_mask]
                            labels_test = data.y[data.test_mask]
                            cls_num = int(labels_test.max()+1)
                        nmi, ari = clustering(H, cls_num, labels_test)
            if epoch % 20 == 0:
                print(f'Pretraining Epoch: {epoch:03d}, loss: {loss:.4f}, best loss: {best_loss:.4f}, nmi: {nmi:.4f}')
        model.load_state_dict(weight)  
        torch.save(model.state_dict(), save_path)
    model.eval()
    H = model.embedding(data)
    cluster_idx = get_clu_idx(H, args.syn_num)
    return model, cluster_idx






def model_training_SSL(args, data, data_test, model, ccenter, clu_idx, lr, epochs, eva=False):

    ccenter.data = ccenter.data.to(args.device)
    optimizer = torch.optim.Adam(list(model.parameters())+[ccenter], lr=lr, weight_decay=args.weight_decay)
    labels_test = data.y[data.test_mask] if args.dataset_name not in ['flickr', 'reddit'] else data_test.y

    best_loss = 1e10
    nmi=0.
    for epoch in range(1, epochs):
        model.train()

        H = model.embedding(data)
        H_norm = F.normalize(H, p=2, dim=-1)
        cc_norm = F.normalize(ccenter, p=2, dim=-1)
        loss1 = SSL_contrast(H_norm, cc_norm, clu_idx)
        loss2 = CC_contrast(cc_norm)
        loss = loss1 + args.alpha * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1):
                nmi_c = 0.0
            else:
                if args.dataset_name in ['flickr', 'reddit']:
                    if model.__class__.__name__ == "GCN":
                        labels_test = data_test.y
                        H_test = model.embedding(data_test)
                        H_test_norm = F.normalize(H_test, p=2, dim=-1)
                        nmi_c, _ = clustering_learn(H_test_norm, cc_norm, labels_test)
                    else:
                        labels_test = data.y
                        nmi_c, _ = clustering_learn(H_norm, cc_norm, labels_test)
                else:
                    nmi_c, _ = clustering_learn(H_norm[data.test_mask], cc_norm, labels_test)

        if loss < best_loss:
            best_loss = loss
            weight = model.state_dict()
            
            if eva == True:
                with torch.no_grad():
                    model.eval()
                    if args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1):
                        nmi_c = 0.0
                    else:
                        if args.dataset_name in ['flickr', 'reddit']:
                            H = model.embedding(data_test)
                            labels_test = data_test.y
                            cls_num = int(labels_test.max()+1)
                        else:
                            H = model.embedding(data)[data.test_mask]
                            labels_test = data.y[data.test_mask]
                            cls_num = int(labels_test.max()+1)
                        nmi, _ = clustering(H, cls_num, labels_test)
            
            if epoch % 1 == 0:
                print(f'SSL Epoch: {epoch:03d}, loss: {loss:.4f}, best loss: {best_loss:.4f}, nmi kmeans: {nmi:.4f}, nmi contrast:{nmi_c:.4f}')
    print()
    model.load_state_dict(weight)
    model.eval()
    H = model.embedding(data)
    cluster_idx = get_clu_idx_cc(H, ccenter)

    return model, cluster_idx, ccenter

def SSL_contrast(H, cluster_center, cluster_idx, temperature=0.3):

    row=np.arange(len(H))
    col=cluster_idx
    pos_mask = np.zeros([len(H),len(cluster_center)], dtype=bool)
    pos_mask[row, col] = True 

    rng= np.random.default_rng()
    neg_num=3*len(cluster_idx)
    neg_mask = np.zeros([len(H),len(cluster_center)], dtype=bool)
    idx = rng.choice(neg_mask.size, neg_num, replace=False)
    neg_mask.ravel()[idx] = True
    neg_mask[row, col] = False 

    # SimCLR loss
    similarity_matrix = torch.matmul(H, cluster_center.T)
    positives = similarity_matrix[pos_mask]
    negatives = similarity_matrix[neg_mask]

    logits = torch.cat([positives, negatives])
    labels = torch.cat([torch.ones(positives.shape[0]),torch.zeros(negatives.shape[0])]).to(H.device)
    logits = logits / temperature
    contrastive_loss = F.cross_entropy(logits, labels) 
    return contrastive_loss

def CC_contrast(cluster_center, temperature=0.3):
    # SimCLR loss
    similarity_matrix = torch.matmul(cluster_center, cluster_center.T)/ temperature
    size = cluster_center.shape[0]
    labels = torch.arange(size).to(cluster_center.device)
    contrastive_loss = F.cross_entropy(similarity_matrix, labels) 
    return contrastive_loss


def adj_generation(args, data, model_spe, ccenter_spe):

    e = data.e.to(args.device)
    data.u = nn.Parameter(torch.randn(args.syn_num, len(e), device = args.device))
    torch.nn.init.xavier_normal_(data.u)

    labels_syn = ccenter_spe.to(args.device)

    optimizer = torch.optim.Adam([data.u], lr=args.lr_u, weight_decay=args.weight_decay)

    print('Adj generation:')
    best_loss = 1e6
    for epoch in range(1, args.epoch_u+1):
        model_spe.train()
        H = model_spe.embedding(data)
        loss1 = generation_loss(args, H, labels_syn)

        orthog_syn = data.u.T @ data.u
        iden = torch.eye(len(e)).to(args.device)
        loss2 = F.mse_loss(orthog_syn, iden) 

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_loss = loss
            u_best = data.u.clone().detach()

        if epoch % 20 == 0:
            print(f'Adj_gene Epoch: {epoch:03d}, loss: {loss:.4f}, best loss: {best_loss:.4f}')

    print()
    L_syn = u_best @ torch.diag(e) @ u_best.T
    adj_syn = torch.eye(args.syn_num).to(args.device) - L_syn
    adj_syn[adj_syn<0]=0
    print("num nodes:", len(adj_syn), "adj sum:", adj_syn.sum().item(), "sparsity:", ((adj_syn>0).sum()*100/(len(adj_syn)**2)).item())
    return adj_syn

def generation_loss(args, h1, h2, t=1.0):
    if args.loss_generation == 'mse':
        return F.mse_loss(h1, h2)
    else:
        h1 = F.normalize(h1, dim=-1, p=2)
        h2 = F.normalize(h2, dim=-1, p=2)
        logits = torch.mm(h1, h2.t()) / t
        labels = torch.arange(h1.size(0), device=h1.device, dtype=torch.long)
        return 0.5 * F.cross_entropy(logits, labels) + 0.5 * F.cross_entropy(logits.t(), labels)
    

def feat_generation(args, data, model_spa, ccenter_spa, adj_syn):
    x_syn = nn.Parameter(torch.randn(args.syn_num, data.x.size(1), device = args.device))
    torch.nn.init.xavier_normal_(x_syn.data)
    labels_syn = ccenter_spa.to(args.device)

    optimizer = torch.optim.Adam([x_syn], lr=args.lr_x, weight_decay=args.weight_decay)
    edge_index = adj_syn.nonzero().t()
    edge_weight = adj_syn[edge_index[0], edge_index[1]]

    print('Feature generation:')

    best_loss = 1e6
    for epoch in range(1, args.epoch_x+1):
        model_spa.train()
        H = model_spa.conv(x_syn, edge_index, edge_weight)
        loss = generation_loss(args, H, labels_syn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_loss = loss
            x_best = x_syn.detach()

        if epoch % 20 == 0:
            print(f'Feat_gene Epoch: {epoch:03d}, loss: {loss:.4f}, best loss: {best_loss:.4f}')
        
    data_syn = Data(x=x_best, edge_index=edge_index.detach(), edge_weight=edge_weight.detach(), y=labels_syn.detach()) 
    return data_syn


def train_model_syn(args, data):

    model = GCN(data.num_features, args.n_dim, args.num_class, 2, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_downstream, weight_decay=args.weight_decay)
    print("Downstream model training:")
    best_loss = 1e6
    for epoch in range(1, args.epoch_downstream+1):
        if epoch == args.epoch_downstream // 2:
            lr = args.lr_downstream*0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.train()
        output = model.embedding(data)
        loss = downstream_loss(output, data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < best_loss:
            best_loss = loss
            weight = model.state_dict()

        if epoch % 20 == 0:
            print(f'Downstream_train Epoch: {epoch:03d}, loss: {loss:.4f}, best loss: {best_loss:.4f}')
        
    model.load_state_dict(weight)  
    return model


def downstream_loss(h1, h2, t=1.0):
    h1 = F.normalize(h1, dim=-1, p=2)
    h2 = F.normalize(h2, dim=-1, p=2)
    logits = torch.mm(h1, h2.t()) / t
    labels = torch.arange(h1.size(0), device=h1.device, dtype=torch.long)
    return 0.5 * F.cross_entropy(logits, labels) + 0.5 * F.cross_entropy(logits.t(), labels)