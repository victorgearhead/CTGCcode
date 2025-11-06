from util.utils import *
from util.module import *
from util.models import *
import cupy as cp
from cupyx.scipy.sparse.linalg import eigsh

def get_dataset(args):

    if args.dataset_name in ["cora"]:
        dataset = Planetoid(args.data_dir, 'cora')
        data = dataset[0]
        data.train_num_original = int(data.train_mask.sum())

    elif args.dataset_name in ['citeseer']:
        dataset = Planetoid(args.data_dir, 'citeseer')
        data = dataset[0]
        data.train_num_original = int(data.train_mask.sum())

    elif args.dataset_name == "ogbn-arxiv":
        dataset_str=args.data_dir+args.dataset_name + "/raw/"
        # adj
        adj_full = sp.load_npz(dataset_str+'adj_full.npz')
        nnodes = adj_full.shape[0]

        adj_full = adj_full + adj_full.T
        adj_full[adj_full > 1] = 1

        # split
        role = json.load(open(dataset_str+'role.json','r'))
        idx_train = role['tr']
        idx_test = role['te']
        idx_val = role['va']
        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        val_mask = torch.zeros(nnodes, dtype=torch.bool)
        test_mask = torch.zeros(nnodes, dtype=torch.bool)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        # label
        class_map = json.load(open(dataset_str + 'class_map.json','r'))
        labels = process_labels(class_map, nnodes)

        # feat
        feat = np.load(dataset_str+'feats.npy')
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)

        dataset = Data(x=torch.FloatTensor(feat).float(), 
                        edge_index=torch.LongTensor(np.array(adj_full.nonzero())), 
                        y=torch.LongTensor(labels), 
                        train_mask = train_mask,  
                        val_mask = val_mask, 
                        test_mask = test_mask)
        transform = T.ToUndirected()
        data = transform(dataset)
        data.train_num_original = int(data.train_mask.sum())

    elif args.dataset_name == "reddit":
        dataset_str=args.data_dir+args.dataset_name + "/raw/"
        # adj
        adj_full = sp.load_npz(dataset_str+'adj_full.npz')
        nnodes = adj_full.shape[0]

        # split
        role = json.load(open(dataset_str+'role.json','r'))
        idx_train = role['tr']
        idx_test = role['te']
        idx_val = role['va']
        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        val_mask = torch.zeros(nnodes, dtype=torch.bool)
        test_mask = torch.zeros(nnodes, dtype=torch.bool)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        # label
        class_map = json.load(open(dataset_str + 'class_map.json','r'))
        labels = process_labels(class_map, nnodes)

        # feat
        feat = np.load(dataset_str+'feats.npy')
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)
        
        dataset = Data(x=torch.FloatTensor(feat).float(), 
                        edge_index=torch.LongTensor(np.array(adj_full.nonzero())), 
                        y=torch.LongTensor(labels), 
                        train_mask = train_mask,  
                        val_mask = val_mask, 
                        test_mask = test_mask)
        transform = T.ToUndirected()
        dataset = transform(dataset)
        data = inductive_processing(dataset)
        data[0].train_num_original = int(data[0].train_mask.sum())

    elif args.dataset_name == "ppi":
        dataset_str = args.data_dir + args.dataset_name + "/raw/"

        # Load adjacency
        adj_full = sp.load_npz(dataset_str + 'adj_full.npz')
        nnodes = adj_full.shape[0]

        # Make undirected
        adj_full = adj_full + adj_full.T
        adj_full[adj_full > 1] = 1

        # Load split indices
        role = json.load(open(dataset_str + 'role.json','r'))
        idx_train = role['tr']
        idx_test  = role['te']
        idx_val   = role['va']

        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        val_mask   = torch.zeros(nnodes, dtype=torch.bool)
        test_mask  = torch.zeros(nnodes, dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val]     = True
        test_mask[idx_test]   = True

        # Load labels (multi-label)
        class_map = json.load(open(dataset_str + 'class_map.json','r'))
        labels = process_labels(class_map, nnodes)

        # Load + normalize features
        feat = np.load(dataset_str + 'feats.npy')
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)

        data = Data(
            x=torch.FloatTensor(feat).float(),
            edge_index=torch.LongTensor(np.array(adj_full.nonzero())),
            y=torch.FloatTensor(labels),         # ✅ FLOAT for multi-label
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        transform = T.ToUndirected()
        data = transform(data)
        data.train_num_original = int(data.train_mask.sum())

    ## pre-processing
    if args.dataset_name != "ppi":
        data = shot_labels(args, data)
    data = link_split_sample(args, data)
    return data

def inductive_processing(data):

    edge_index,_ = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)
    # edge_index = SparseTensor.from_edge_index(edge_index)
    x = data.x[data.train_mask]
    y = data.y[data.train_mask]
    g_train = Data(x=x, y=y, edge_index=edge_index)
    g_train.train_mask = torch.ones(len(x), dtype=torch.bool)

    edge_index,_ = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)
    # edge_index = SparseTensor.from_edge_index(edge_index)
    x = data.x[data.val_mask]
    y = data.y[data.val_mask]
    g_val = Data(x=x, y=y, edge_index=edge_index)
    g_val.val_mask = torch.ones(len(x), dtype=torch.bool)

    edge_index,_ = subgraph(data.test_mask, data.edge_index, relabel_nodes=True)
    # edge_index = SparseTensor.from_edge_index(edge_index)
    x = data.x[data.test_mask]
    y = data.y[data.test_mask]
    g_test = Data(x=x, y=y, edge_index=edge_index)
    g_test.test_mask = torch.ones(len(x), dtype=torch.bool)

    return [g_train, g_val, g_test]


def set_dataset(args, datasets):
    if args.dataset_name in ['flickr', 'reddit']:
        data, data_val, data_test = datasets
        data = tranverse_adj(data)
        data_val = tranverse_adj(data_val)
        data_test = tranverse_adj(data_test)
        data, data_val, data_test = data.to(args.device), data_val.to(args.device), data_test.to(args.device)
    elif args.dataset_name in ['ogbn-products']:
        data = to_device(datasets, args.device)
        data = tranverse_adj(data)
        data_val, data_test = None, None
    else:
        data = datasets.to(args.device)
        data = tranverse_adj(data)
        data_val, data_test = None, None
    args.num_class = int(data.y.max()+1)
    return args, data, data_val, data_test 

def to_device(datasets, device):
    datasets.x = datasets.x.to(device)
    datasets.edge_index = datasets.edge_index.to(device)
    datasets.y = datasets.y.to(device)
    datasets.train_mask = datasets.train_mask.to(device)
    datasets.val_mask = datasets.val_mask.to(device)
    datasets.test_mask = datasets.test_mask.to(device)
    return datasets

def process_labels(class_map, nnodes):
    """
    setup vertex property map for output classests
    """
    num_vertices = nnodes
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        nclass = num_classes
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[int(k)] = v
    else:
        class_arr = np.zeros(num_vertices, dtype=int)
        for k, v in class_map.items():
            class_arr[int(k)] = v
        class_arr = class_arr - class_arr.min()
        nclass = max(class_arr) + 1
    return class_arr


def shot_labels(args, data):
    file_path = args.split_data_dir+f'{args.dataset_name}_label_shot.pkl'
    if args.dataset_name in ['cora', 'citeseer', 'ogbn-arxiv', 'ogbn-products','amazon', 'ppi']:
        nnodes = data.x.shape[0]

        try:
            with open(file_path, 'rb') as f:
                labels = pickle.load(f)
        except:  
            print('generate splite')
            train_mask = torch.zeros(nnodes, dtype=torch.bool)
            idx = torch.arange(nnodes)
            idx_train = idx[data.train_mask]
            label_train = data.y[data.train_mask]
            labels = {}
            for shot in [1,3,5]:
                labels[shot] = []
                for repeat in range(5):
                    sel_train = []
                    for cls_num in range(data.y.max()+1):
                        idx_cls = label_train==cls_num
                        buf = idx_train[idx_cls]
                        selected_train = np.random.choice(buf, shot, replace=False) if buf.shape[0]>shot else np.array(buf)
                        sel_train.append(selected_train)
                    labels[shot].append(np.concatenate(sel_train))
            with open(file_path, 'wb') as f:
                pickle.dump(labels, f)    

        idx_train = labels[args.shot][args.seed//5]
        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        train_mask[idx_train] = True
        data.train_mask = train_mask
        print('shot:', args.shot, 'num_class', int(data.y.max()+1), 'num_train', int(data.train_mask.sum()))

    else:
        data_train = data[0]
        nnodes = data_train.x.shape[0]
        try:
            with open(file_path, 'rb') as f:
                labels = pickle.load(f)
        except:  
            print('generate splite')
            train_mask = torch.zeros(nnodes, dtype=torch.bool)
            idx = torch.arange(nnodes)
            idx_train = idx[data_train.train_mask]
            label_train = data_train.y[data_train.train_mask]
            labels = {}
            for shot in [1,3,5]:
                labels[shot] = []
                for repeat in range(5):
                    sel_train = []
                    for cls_num in range(data_train.y.max()+1):
                        idx_cls = label_train==cls_num
                        buf = idx_train[idx_cls]
                        selected_train = np.random.choice(buf, shot, replace=False) if buf.shape[0]>shot else np.array(buf)
                        sel_train.append(selected_train)
                    labels[shot].append(np.concatenate(sel_train))
            with open(file_path, 'wb') as f:
                pickle.dump(labels, f)       

        idx_train = labels[args.shot][args.seed//5]
        train_mask = torch.zeros(nnodes, dtype=torch.bool)
        train_mask[idx_train] = True
        data_train.train_mask = train_mask
        print('shot:', args.shot, 'num_class', int(data_train.y.max()+1), 'num_train', int(data_train.train_mask.sum()))
        data = [data_train, data[1], data[2]]

    return data


def link_split_sample(args, data):
    file_path = args.split_data_dir+f'{args.dataset_name}_links_sample.pt'
    if args.dataset_name in ['cora', 'citeseer', 'ogbn-arxiv', 'ogbn-products','amazon', 'ppi']:
        try:
            links = torch.load(file_path)
            train_edge_index, train_edge_label, train_edge_label_index, \
            val_edge_index, val_edge_label, val_edge_label_index, \
            test_edge_index, test_edge_label, test_edge_label_index = links
        except:    
            print('generate splite')
            transform = T.RandomLinkSplit(num_val=0.05, num_test=0.15, is_undirected=True, add_negative_train_samples=False)
            train_data, val_data, test_data = transform(data)

            train_edge_index = train_data.edge_index
            train_edge_label = train_data.edge_label
            train_edge_label_index = train_data.edge_label_index

            val_edge_index = val_data.edge_index
            val_edge_label = val_data.edge_label
            val_edge_label_index = val_data.edge_label_index

            test_edge_index = test_data.edge_index
            test_edge_label = test_data.edge_label
            test_edge_label_index = test_data.edge_label_index

            num_edge = len(train_edge_label)
            idx = np.random.choice(np.arange(0, num_edge), 100, replace = False)
            train_edge_label = train_edge_label[idx]
            train_edge_label_index = train_edge_label_index[:,idx] 

            if args.dataset_name in ['ogbn-arxiv']: # sampling the train/val/test edges 

                num_edge = len(val_edge_label)
                idx = np.random.choice(np.arange(0, num_edge), int(num_edge/100), replace = False)
                val_edge_label = val_edge_label[idx]
                val_edge_label_index = val_edge_label_index[:,idx] 

                num_edge = len(test_edge_label)
                idx = np.random.choice(np.arange(0, num_edge), int(num_edge/100), replace = False)
                test_edge_label = test_edge_label[idx]
                test_edge_label_index = test_edge_label_index[:,idx] 

            if args.dataset_name in ['ogbn-products']: # sampling the train/val/test edges 

                num_edge = len(val_edge_label)
                idx = np.random.choice(np.arange(0, num_edge), int(num_edge/1000), replace = False)
                val_edge_label = val_edge_label[idx]
                val_edge_label_index = val_edge_label_index[:,idx] 

                num_edge = len(test_edge_label)
                idx = np.random.choice(np.arange(0, num_edge), int(num_edge/3000), replace = False)
                test_edge_label = test_edge_label[idx]
                test_edge_label_index = test_edge_label_index[:,idx] 

            torch.save([train_edge_index, train_edge_label, train_edge_label_index, 
                        val_edge_index, val_edge_label, val_edge_label_index, 
                        test_edge_index, test_edge_label, test_edge_label_index], file_path)      

        data.edge_index = train_edge_index # training with the edge_index, not train_edge_index
        data.train_edge_label = train_edge_label
        data.train_edge_label_index = train_edge_label_index

        data.val_edge_index = val_edge_index
        data.val_edge_label = val_edge_label # added edges are asymmetric
        data.val_edge_label_index = val_edge_label_index

        data.test_edge_index = test_edge_index
        data.test_edge_label = test_edge_label
        data.test_edge_label_index = test_edge_label_index

    else:
        train_data, val_data, test_data = data

        try:
            links = torch.load(file_path)
            train_edge_index, train_edge_label, train_edge_label_index, \
            val_edge_index, val_edge_label, val_edge_label_index, \
            test_edge_index, test_edge_label, test_edge_label_index = links

        except:    
            print('generate splite')

            transform = T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True, add_negative_train_samples=False)
            train_d = transform(train_data)[0]

            transform = T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True, add_negative_train_samples=True)
            val_d = transform(val_data)[0]

            transform = T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True, add_negative_train_samples=True)
            test_d = transform(test_data)[0]

            train_edge_index = train_d.edge_index
            train_edge_label = train_d.edge_label
            train_edge_label_index = train_d.edge_label_index

            val_edge_index = val_d.edge_index
            val_edge_label = val_d.edge_label
            val_edge_label_index = val_d.edge_label_index

            test_edge_index = test_d.edge_index
            test_edge_label = test_d.edge_label
            test_edge_label_index = test_d.edge_label_index

            # sampling the train/val/test edges 
            
            ratio1 = 100 if args.dataset_name in ['reddit'] else 2
            ratio2 = 100 if args.dataset_name in ['reddit'] else 2
            ratio3 = 500 if args.dataset_name in ['reddit'] else 2

            num_edge = len(train_edge_label)
            idx = np.random.choice(np.arange(0, num_edge), 100, replace = False)
            train_edge_label = train_edge_label[idx]
            train_edge_label_index = train_edge_label_index[:,idx] 

            num_edge = len(val_edge_label)
            idx = np.random.choice(np.arange(0, num_edge), int(num_edge/ratio2), replace = False)
            val_edge_label = val_edge_label[idx]
            val_edge_label_index = val_edge_label_index[:,idx] 

            num_edge = len(test_edge_label)
            idx = np.random.choice(np.arange(0, num_edge), int(num_edge/ratio3), replace = False)
            test_edge_label = test_edge_label[idx]
            test_edge_label_index = test_edge_label_index[:,idx] 

            torch.save([train_edge_index, train_edge_label, train_edge_label_index, 
                        val_edge_index, val_edge_label, val_edge_label_index, 
                        test_edge_index, test_edge_label, test_edge_label_index], file_path)  


        train_data.edge_index = train_edge_index # training with the edge_index, not train_edge_index
        train_data.train_edge_label = train_edge_label
        train_data.train_edge_label_index = train_edge_label_index

        val_data.val_edge_index = val_edge_index
        val_data.val_edge_label = val_edge_label
        val_data.val_edge_label_index = val_edge_label_index

        test_data.test_edge_index = test_edge_index
        test_data.test_edge_label = test_edge_label
        test_data.test_edge_label_index = test_edge_label_index

        data = [train_data, val_data, test_data]
    print('split links for LP')
    return data

def tranverse_adj(data):
    edge_index = data.edge_index
    adj_t = SparseTensor(row = edge_index[0], col=edge_index[1], sparse_sizes=(data.num_nodes,data.num_nodes))
    data.edge_index = adj_t
    return data


def aug_full_connected(x, adj, num_nodes):
    try:
        mx = adj.coo()
        edge_index = torch.vstack([mx[0], mx[1]])  
    except:
        mx = adj
        edge_index = torch.vstack([mx[0], mx[1]]) 

    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    nx_graph = convert.to_networkx(data, to_undirected=True)
    
    components = sorted(nx.connected_components(nx_graph), key=len, reverse=True)
    largest_cc = np.array(list(components[0]))
    edge_added = []
    print("link components")
    idx = np.random.choice(largest_cc, 1000, replace=False)
    x_lcc = x[idx]
    x_lcc = F.normalize(x_lcc, p=2, dim=-1)

    for c in components[1:]:
        source = list(c)[0] 
        x_src = x[source]
        x_src = F.normalize(x_src, p=2, dim=-1)
        similarity = torch.matmul(x_src, x_lcc.T)
        idx_max = torch.argmax(similarity)
        target = idx[idx_max]
        edge_added.append((source, target))
        edge_added.append((target, source))

    nx_graph.add_edges_from(edge_added)

    largest_cc = max(nx.connected_components(nx_graph), key=len)
    idx_lcc = list(largest_cc)
    largest_cc_graph = nx_graph.subgraph(largest_cc)
    print("norm adj")
    if len(idx_lcc) == num_nodes:
        adj_lcc = nx.to_scipy_sparse_array(largest_cc_graph)
        adj_norm_lcc = normalize_adj(adj_lcc)
    else:
        adj_lcc = nx.to_scipy_sparse_array(largest_cc_graph)
        adj_norm_lcc = normalize_adj(adj_lcc)
    print("norm finished")    
    L_lcc = sp.eye(len(idx_lcc)) - adj_norm_lcc    
    return L_lcc


def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    
    return mx

def get_eigens(args, laplacian_matrix, save=True):
    data_name = args.dataset_name

    if data_name in [ 'ogbn-arxiv','ogbn-products', 'flickr', 'reddit', 'twitch-gamer', 'ppi']:
        print("SA eigsh calculation")
        laplacian_matrix_gpu = cp.sparse.csr_matrix(laplacian_matrix)
        eigenvalues, eigenvectors = eigsh(laplacian_matrix_gpu, k=1000, which="SA", tol=1e-5)        
    else:
        print("eigsh calculation")
        if sp.issparse(laplacian_matrix):
            laplacian_matrix = laplacian_matrix.todense()
        eigenvalues, eigenvectors = eigh(laplacian_matrix)
    
    if data_name in [ 'ogbn-arxiv','ogbn-products', 'flickr', 'reddit', 'twitch-gamer', 'ppi']:
        print("LA eigsh calculation")
        laplacian_matrix_gpu = cp.sparse.csr_matrix(laplacian_matrix)
        eigenvalues_la, eigenvectors_la = eigsh(laplacian_matrix_gpu, k=1000, which="LA", tol=1e-5)
        eigenvalues = np.hstack([eigenvalues, eigenvalues_la])
        eigenvectors = np.hstack([eigenvectors, eigenvectors_la])
        
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx[:]]
    eigenvectors = eigenvectors[:, idx[:]]
                     
    return eigenvalues, eigenvectors

def load_eigens(args, data):

    args.eigvals_path = args.eigen_path+ "eigenvalues.npy"
    args.eigvecs_path = args.eigen_path+ "eigenvectors.npy"

    eigenvals_lcc = torch.FloatTensor(np.load(args.eigvals_path))
    eigenvecs_lcc = torch.FloatTensor(np.load(args.eigvecs_path))

    args.eigen_k = args.syn_num

    k1 = math.ceil(args.eigen_k * 0.9)
    k2 = args.eigen_k - k1
    print("k1:", k1, ",", "k2:", k2)
    k1_end = (k1 - 1) + 1
    eigen_sum = eigenvals_lcc.shape[0]
    k2_end = eigen_sum - (k2 - 1) - 1
    k1_list = np.arange(0, k1_end, 1)
    k2_list = np.arange(k2_end, eigen_sum, 1)
    eigenvals = torch.cat([eigenvals_lcc[k1_list], eigenvals_lcc[k2_list]])
    eigenvecs = torch.cat([eigenvecs_lcc[:, k1_list], eigenvecs_lcc[:, k2_list]], dim=1)

    data.e = eigenvals.to(args.device)
    data.u = eigenvecs.to(args.device)
    return data


def load_pre_train(args, model_spa, model_spe):
    save_path = args.model_path+f'{args.dataset_name}_rate_{args.reduction_rate}_seed_{args.seed}.pth'

    data = torch.load(save_path)
    model_spa_ = data['model_spa']
    model_spe_ = data['model_spe']
    model_spa.load_state_dict(model_spa_)
    model_spe.load_state_dict(model_spe_)

    ccenter_spa = data['ccenter_spa'].to(args.device)
    ccenter_spe = data['ccenter_spe'].to(args.device)

    return model_spa, ccenter_spa, model_spe, ccenter_spe