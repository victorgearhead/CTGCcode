from util.utils import *
from util.module import *
from util.models import *



def pre_train(args, data, data_test, model, load=True):
    save_path = args.model_path + args.dataset_name + "_pre.pt"
    if os.path.exists(save_path):
        param = torch.load(save_path)
        model_dict = model.state_dict()
        filtered_param = {k: v for k, v in param.items() if 'classifier' not in k}
        model_dict.update(filtered_param)
        model.load_state_dict(model_dict)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        n = data.x.shape[0]
        disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0).to(args.device)

        best_loss = 1e6
        nmi = 0.
        weight = model.state_dict()
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
                    if args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1):
                        nmi, ari = 0., 0.
                    else:
                        if args.dataset_name in ['flickr', 'reddit']:
                            H = model.embedding(data_test)
                            labels_test = data_test.y
                            cls_num = int(labels_test.max() + 1)
                        else:
                            H = model.embedding(data)[data.test_mask]
                            labels_test = data.y[data.test_mask]
                            cls_num = int(labels_test.max() + 1)
                        nmi, ari = clustering(H, cls_num, labels_test)
            if epoch % 20 == 0:
                print(f'Pretraining Epoch: {epoch:03d}, loss: {loss:.4f}, best loss: {best_loss:.4f}, nmi: {nmi:.4f}')
        model.load_state_dict(weight)
        torch.save(model.state_dict(), save_path)

    model.eval()
    H_all = model.embedding(data) 
    H_train = H_all[data.train_mask]
    n_train = H_train.shape[0]

    try:
        num_classes = int(data.y.max().item() + 1)
    except:
        num_classes = 1

    requested_K = int(max(1, int(round(data.train_num_original * getattr(args, 'reduction_rate', 0.1)))))
    K_train = min(max(num_classes, requested_K), max(1, n_train))
    if n_train <= 0:
        raise RuntimeError("No training nodes found (data.train_mask is empty).")

    cluster_idx_train, K_actual = get_clu_idx(H_train, K_train)
    
    ccentroids_train = compute_centroids_from_indices(H_train, cluster_idx_train, K_actual)

    cluster_idx_full = assign_to_centroids(H_all, ccentroids_train)

    membership = {}
    train_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1).cpu().numpy()
    cluster_idx_train_np = cluster_idx_train if isinstance(cluster_idx_train, np.ndarray) else cluster_idx_train.cpu().numpy()
    for k in range(K_actual):
        members = train_idx[cluster_idx_train_np == k].tolist()
        membership[k] = members
    save_json(membership, os.path.join(args.result_path, f"{args.dataset_name}_cluster_membership_seed{args.seed}.json"))

    return model, cluster_idx_full
def CC_contrast(cluster_center, temperature=0.3):
    
    similarity_matrix = torch.matmul(cluster_center, cluster_center.T)/ temperature
    size = cluster_center.shape[0]
    labels = torch.arange(size).to(cluster_center.device)
    contrastive_loss = F.cross_entropy(similarity_matrix, labels) 
    return contrastive_loss

def model_training_SSL(args, data, model, ccenter, clu_idx, lr, epochs, eva=False):

    if hasattr(ccenter, 'data'):
        ccenter.data = ccenter.data.to(args.device)
    else:
        ccenter = ccenter.to(args.device)

    optimizer = torch.optim.Adam(list(model.parameters()) + [ccenter], lr=lr, weight_decay=args.weight_decay)

    train_mask = data.train_mask
    y_train = data.y[train_mask] if args.dataset_name not in ['flickr', 'reddit'] else data.y[train_mask]

    best_loss = 1e10
    nmi = 0.

    if torch.is_tensor(clu_idx):
        clu_idx = clu_idx.cpu().numpy()
    else:
        clu_idx = np.array(clu_idx)

    for epoch in range(1, epochs):
        model.train()
        H_all = model.embedding(data)         
        H_norm = F.normalize(H_all, p=2, dim=-1)
        cc_norm = F.normalize(ccenter, p=2, dim=-1)

        train_indices = np.nonzero(train_mask.cpu().numpy())[0]
        H_train_norm = H_norm[train_mask]

        clu_idx_train = clu_idx[train_indices]

        loss1 = SSL_contrast_train(H_train_norm, cc_norm, clu_idx_train) 
        loss2 = CC_contrast(cc_norm)
        loss = loss1 + args.alpha * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1):
                nmi_c = 0.0
            else:
                H_train_for_clust = H_train_norm.detach().cpu()
                cc_for_clust = cc_norm.detach().cpu()
                nmi_c, _ = clustering_learn(H_train_for_clust, cc_for_clust, y_train.cpu().numpy())

        if loss < best_loss:
            best_loss = loss
            weight = model.state_dict()
            nmi = nmi_c

        if epoch % 20 == 0:
            print(f'SSL Epoch: {epoch:03d}, loss: {loss:.4f}, best loss: {best_loss:.4f}, nmi contrast(train):{nmi_c:.4f}')

    model.load_state_dict(weight)
    model.eval()

    H_all = model.embedding(data).detach()
    H_train = H_all[train_mask]    
    train_indices = np.nonzero(train_mask.cpu().numpy())[0]
    clu_idx_train = np.array(clu_idx)[train_indices]
    if len(clu_idx_train) == 0:
        K_actual = 1
    else:
        K_actual = int(np.max(clu_idx_train) + 1)

    ccenters = compute_centroids_from_indices(H_train, clu_idx_train, K_actual).to(args.device)

    cluster_idx_full = assign_to_centroids(H_all, ccenters)

    return model, cluster_idx_full, ccenters


def SSL_contrast_train(H_train_norm, cluster_center, cluster_idx_train, temperature=0.3):
    N = H_train_norm.shape[0]
    K = cluster_center.shape[0]
    pos_sim = (H_train_norm * cluster_center[cluster_idx_train]).sum(dim=1)
    all_sim = torch.matmul(H_train_norm, cluster_center.t()) 
    pos_mask = torch.zeros_like(all_sim, dtype=torch.bool)
    rows = torch.arange(N, device=H_train_norm.device)
    pos_mask[rows, torch.tensor(cluster_idx_train, device=H_train_norm.device)] = True
    negatives = all_sim[~pos_mask].view(N, K-1)
    logits = torch.cat([pos_sim.unsqueeze(1), negatives], dim=1) / temperature
    labels = torch.zeros(N, dtype=torch.long, device=H_train_norm.device)
    loss = F.cross_entropy(logits, labels)
    return loss


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


def downstream_loss(logits, labels):
    if labels.dtype in (torch.float32, torch.float64):
        logits = F.normalize(logits, p=2, dim=-1)
        labels = F.normalize(labels, p=2, dim=-1)
        return 1 - F.cosine_similarity(logits, labels).mean()
    else:
        return F.cross_entropy(logits, labels)
