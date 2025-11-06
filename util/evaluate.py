from util.utils import *
from util.module import *
from util.models import *



def eva_data(args, data, data_val, data_test, model):
    with torch.no_grad():
        model.eval()
        if args.dataset_name in ['flickr', 'reddit']:
            H = model.embedding(data)
            H_val = model.embedding(data_val)
            H_test = model.embedding(data_test)  
            H_test_masked = H_test
            labels_test = data_test.y
        else:
            H = model.conv(data.x, data.edge_index)
            H_val = model.conv(data.x, data.val_edge_index)
            H_test = model.conv(data.x, data.test_edge_index)
            H_test_masked = H_test[data.test_mask]
            labels_test = data.y[data.test_mask]

        if args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1):
            return H, H_val, H_test, H_test_masked, labels_test, None, None, None, None
        
        H_train_shot_3, label_train_shot_3 = get_shot_train(args, H, data.y, shot=3)
        H_train_shot_5, label_train_shot_5 = get_shot_train(args, H, data.y, shot=5)
    return H, H_val, H_test, H_test_masked, labels_test, H_train_shot_3, label_train_shot_3, H_train_shot_5, label_train_shot_5


def get_shot_train(args, H, label, shot):
    nnodes=len(H)
    file_path = args.split_data_dir+f'{args.dataset_name}_label_shot.pkl'
    with open(file_path, 'rb') as f:
        labels = pickle.load(f) 
    if args.dataset_name in ['cora', 'citeseer', 'ogbn-arxiv', 'ogbn-products','amazon']:
        idx_train = labels[shot][args.seed//5]
        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        train_mask[idx_train] = True
    else:
        idx_train = labels[shot][args.seed//5]
        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        train_mask[idx_train] = True
    return H[train_mask], label[train_mask]



def evaluate_CL(H_test, labels_test):
    cls_num = int(labels_test.max()+1)
    nmi, ari = clustering(H_test, cls_num, labels_test)
    print(f'NMI Test: {nmi:.4f}')
    print(f'ARI Test: {ari:.4f}')
    return nmi, ari


def evaluate_NC(args, H_train_shot_3, label_train_shot_3, H_train_shot_5, label_train_shot_5, H_test, labels_test):
    acc = []
    print('train node classification task')
    for iter in [3,5]:
        if iter ==3:
            H_train = H_train_shot_3
            label_train = label_train_shot_3
        else:
            H_train = H_train_shot_5
            label_train = label_train_shot_5

        classifier = LogReg(args.n_dim, args.num_class).to(args.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        ## train
        test_acc = 0
        best_loss = 1e6
        for epoch in range(1, args.epoch_cls):
            if epoch == args.epoch_cls // 2:
                lr = args.lr_cls*0.1
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=lr, weight_decay=args.weight_decay)

            classifier.train()
            output = classifier(H_train)
            loss = criterion(output, label_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                classifier.eval()
                pred = classifier(H_test).argmax(1)
                tmp_test_acc = pred.eq(labels_test).sum().item() / len(labels_test)

            if loss < best_loss:
                best_loss = loss
                test_acc = tmp_test_acc

            if epoch % 1 == 0:
                print(f'NC Epoch: {epoch:03d}, Test: {tmp_test_acc:.4f}, Best Test: {test_acc:.4f}')
        acc.append(test_acc)
    print(f'NC ACC Test shot 3: {acc[0]:.4f}', f'shot 5: {acc[1]:.4f}')
    return acc


def evaluate_LP(args, data, H, H_val, H_test, data_val, data_test):
    
    print('train link prediction task')
    linkpredicter = LogReg(args.n_dim, args.n_dim).to(args.device)
    optimizer = torch.optim.Adam(linkpredicter.parameters(), lr=args.lr_lp, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    neg_edge_index = train_negative_sampling(data)

    ## train
    best_val_acc = test_acc = 0
    for epoch in range(1, args.epoch_lp):
        if epoch == args.epoch_lp // 2:
            lr = args.lr_lp*0.1
            optimizer = torch.optim.Adam(linkpredicter.parameters(), lr=lr, weight_decay=args.weight_decay)

        linkpredicter.train()
        logit = linkpredicter(H)
        edge_label_index, edge_label = edge_negative_sampling(data, neg_edge_index)
        out = link_pre(logit, edge_label_index)
        loss = criterion(out, edge_label) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.dataset_name in ['flickr', 'reddit']:
            val_acc, val_auc, tmp_test_acc, tmp_test_auc = pre_link(H_val, data_val, H_test, data_test, linkpredicter)
        else:
            val_acc, val_auc, tmp_test_acc, tmp_test_auc = pre_link(H_val, data, H_test, data, linkpredicter)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            test_auc = tmp_test_auc

        if epoch % 20 == 0:
            print(f'LP Epoch: {epoch:03d}, Val: {val_acc:.4f}, Test ACC: {test_acc:.4f}, Test AUC: {test_auc:.4f}')
    
    print(f'LP AUC Test: {test_auc:.4f}')
    print(f'LP ACC Test: {test_acc:.4f}')

    return test_auc, test_acc


def train_negative_sampling(train_data):
    # We perform a new round of negative sampling for every training epoch:
    row, col, _ = train_data.edge_index.coo()
    edge_index = torch.stack([row, col], dim=0).long()
    num_neg_samples = 10000 if edge_index.size(1)>10000 else edge_index.size(1)
    neg_edge_index = negative_sampling(
        edge_index=edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=num_neg_samples, method='sparse')

    return neg_edge_index

def edge_negative_sampling(train_data, neg_edge_index):
    # We perform a new round of negative sampling for every training epoch:
    neg_num = len(train_data.train_edge_label)*3
    idx = np.random.choice(neg_edge_index.size(1), neg_num, replace=False)
    neg_edge_index_sampled = neg_edge_index[:,idx]
    edge_label_index = torch.cat(
        [train_data.train_edge_label_index, neg_edge_index_sampled],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.train_edge_label,
        train_data.train_edge_label.new_zeros(neg_num)
    ], dim=0)

    return edge_label_index, edge_label


def link_pre(x, edge_label_index):
    return (x[edge_label_index[0]] * x[edge_label_index[1]]).sum(dim=-1).view(-1)

def pre_link(H_val, data_val, H_test, data_test, linkpredicter):
    with torch.no_grad():
        linkpredicter.eval() 
        logits = linkpredicter(H_val)
        out = link_pre(logits, data_val.val_edge_label_index).sigmoid()
        auc_val = roc_auc_score(data_val.val_edge_label.cpu().numpy(), out.cpu().numpy())
        pre = (out.cpu().numpy()>0.5).astype(np.int64)
        acc_val = accuracy_score(data_val.val_edge_label.cpu().numpy(), pre)

        logits = linkpredicter(H_test)
        out = link_pre(logits, data_test.test_edge_label_index).sigmoid()
        auc_test = roc_auc_score(data_test.test_edge_label.cpu().numpy(), out.cpu().numpy())
        pre = (out.cpu().numpy()>0.5).astype(np.int64)
        acc_test = accuracy_score(data_test.test_edge_label.cpu().numpy(), pre)
    return acc_val, auc_val, acc_test, auc_test