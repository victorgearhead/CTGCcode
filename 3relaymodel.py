from util.data_loader import *
from util.evaluate import *
from util.hyperpara import *
from util.models import *
from util.module import *
from util.training import *
from util.utils import *
import warnings
import numpy as np
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="cora", help='cora, citeseer, ogbn-arxiv, reddit, Roman-empire')
parser.add_argument('--gpu', type=int, default=4)
parser.add_argument('--result_path', type=str, default="./results/")
parser.add_argument('--model_path', type=str, default="./save_pretrain_model/")
parser.add_argument('--eigen_path', type=str, default="./save_eigen/")
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--split_data_dir', type=str, default="./dataset_split/")

parser.add_argument('--epoch_pretrain', type=int, default=200)
parser.add_argument('--epoch_ssl', type=int, default=20)
parser.add_argument('--iter_num', type=int, default=5)
parser.add_argument('--lr_pretrain', type=float, default=0.001)
parser.add_argument('--lr_ssl_spa', type=float, default=0.0001)
parser.add_argument('--lr_ssl_spe', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=1000)

parser.add_argument('--reduction_rate', type=float, default=0.5)
parser.add_argument('--epoch_cls', type=int, default=200)
parser.add_argument('--epoch_lp', type=int, default=200)
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--lr_lp', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--nrepeat', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_dim', type=int, default=256)
parser.add_argument('--eva_iter', type=int, default=1)
parser.add_argument('--test_gnn', type=str, default='GCN')
parser.add_argument('--shot', type=int, default=3)
args = parser.parse_args()
args = device_setting(args)
seed_everything(args.seed)

args.result_path = f'./results_proposed/'
args.eigen_path += f'{args.dataset_name}/'
os.makedirs(args.result_path, exist_ok=True)
os.makedirs(args.model_path, exist_ok=True)
os.makedirs(args.eigen_path, exist_ok=True)
args = SSL_hyperpara(args)

acc_shot3_NC = []
acc_shot5_NC = []
acc_cluster_NC = [] 
auc_LP = []
acc_LP = []
nmi_CL = []
ari_CL = []

def fewshot_prototype_eval(H_train_shot, y_train_shot, H_test, y_test, device):

    H_train_shot = H_train_shot.to(device)
    H_test = H_test.to(device)
    y_train_shot = y_train_shot.to(device)
    y_test = y_test.to(device)

    classes = torch.unique(y_train_shot)
    d = H_train_shot.size(1)
    num_classes = int(torch.max(y_train_shot).item() + 1) if classes.numel() > 0 else 1
    prototypes = torch.zeros((num_classes, d), device=device)
    for c in range(num_classes):
        mask = (y_train_shot == c)
        if mask.sum() > 0:
            prototypes[c] = H_train_shot[mask].mean(dim=0)
        else:
            prototypes[c] = torch.zeros(d, device=device)

    prototypes = F.normalize(prototypes, p=2, dim=-1)
    H_test_norm = F.normalize(H_test, p=2, dim=-1)

    sims = torch.matmul(H_test_norm, prototypes.T)
    preds = torch.argmax(sims, dim=1)
    acc = (preds == y_test.to(device)).float().mean().item()
    return acc

def cluster_centroid_eval(H_test, y_test, cluster_centroids, device):
    """
    Use centroids (K x d) to classify test nodes by nearest centroid.
    y_test are true labels for test nodes (tensor).
    Returns (accuracy_float)
    """
    H_test = H_test.to(device)
    cluster_centroids = cluster_centroids.to(device)
    H_test_norm = F.normalize(H_test, p=2, dim=-1)
    cluster_norm = F.normalize(cluster_centroids, p=2, dim=-1)
    sims = torch.matmul(H_test_norm, cluster_norm.T)
    preds = torch.argmax(sims, dim=1)
    acc = (preds == y_test.to(device)).float().mean().item()
    return acc

for i in range(args.nrepeat):
    args.seed += 1

    datasets = get_dataset(args)
    args, data, data_val, data_test = set_dataset(args, datasets)
    print("train num:", int(data.train_mask.sum()))

    train_num = int(getattr(data, 'train_num_original', int(data.train_mask.sum())))
    num_classes = int(data.y.max().item() + 1) if data.y.dim() == 1 else data.y.size(1)
    requested = int(train_num * args.reduction_rate)
    args.syn_num = int(min(max(requested, num_classes), max(1, train_num)))
    print(f"syn_num set to {args.syn_num} (requested {requested}, num_classes {num_classes}, train_num {train_num})")

    data = load_eigens(args, data)

    model_spa = GCN(data.num_features, args.n_dim, args.syn_num, 2, args.dropout).to(args.device)
    model_spe = EigenMLP(args.n_dim, args.syn_num, args.n_dim).to(args.device)

    model_spa, cluster_embs = pre_train(args, data, data_val, model_spa)

    if isinstance(cluster_embs, np.ndarray):
        cluster_embs = torch.from_numpy(cluster_embs).float()
    else:
        cluster_embs = cluster_embs.detach().float()
    H_all = model_spa.embedding(data)
    if not isinstance(H_all, torch.Tensor):
        H_all = torch.tensor(H_all, dtype=torch.float32)
    H_all = H_all.detach()

    cluster_idx = assign_to_centroids(H_all, cluster_embs)
    ccenter_spa, ccenter_spe = init_cc(args, data, model_spa, model_spe, cluster_idx)

    for _it in range(args.iter_num):
        model_spa, cluster_idx, ccenter_spa = model_training_SSL(
            args, data, model_spa, ccenter_spa, cluster_idx, args.lr_ssl_spa, args.epoch_ssl
        )
        model_spe, cluster_idx, ccenter_spe = model_training_SSL(
            args, data, model_spe, ccenter_spe, cluster_idx, args.lr_ssl_spe, args.epoch_ssl
        )

    save_pre_train(args, model_spa, ccenter_spa, model_spe, ccenter_spe)

    H, H_val, H_test, H_test_masked, labels_test, \
    H_train_shot_3, label_train_shot_3, \
    H_train_shot_5, label_train_shot_5 = eva_data(args, data, data_val, data_test, model_spa)

    if args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1):
        nmi, ari = 0.0, 0.0
    else:
        nmi, ari = evaluate_CL_labels(H_test_masked, labels_test)

    if args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1):
        acc_shot3, acc_shot5, acc_cluster = None, None, None
    else:
        device = args.device

        acc_shot3 = fewshot_prototype_eval(
            H_train_shot_3.to(device), label_train_shot_3.to(device),
            H_test_masked.to(device), labels_test.to(device), device
        )

        acc_shot5 = fewshot_prototype_eval(
            H_train_shot_5.to(device), label_train_shot_5.to(device),
            H_test_masked.to(device), labels_test.to(device), device
        )

        if isinstance(ccenter_spa, np.ndarray):
            ccenter_spa = torch.from_numpy(ccenter_spa).float()
        else:
            ccenter_spa = ccenter_spa.detach().float()
        acc_cluster = cluster_centroid_eval(H_test_masked, labels_test, ccenter_spa, device)

        print(f"[Eval] Mean cosine similarity (centroid): {evaluate_CL_cosine(H_test_masked, ccenter_spa)[0]:.4f}, "
              f"Few-shot3 acc: {acc_shot3:.4f}, Few-shot5 acc: {acc_shot5:.4f}, Centroid acc: {acc_cluster:.4f}")

    auc_lp, acc_lp = evaluate_LP(args, data, H, H_val, H_test, data_val, data_test)

    if acc_shot3 is not None: acc_shot3_NC.append(acc_shot3)
    if acc_shot5 is not None: acc_shot5_NC.append(acc_shot5)
    if acc_cluster is not None: acc_cluster_NC.append(acc_cluster)
    auc_LP.append(auc_lp)
    acc_LP.append(acc_lp)
    nmi_CL.append(nmi)
    ari_CL.append(ari)
    print()

teacher_record_caption(args)

if args.dataset_name != "ppi":
    if len(acc_shot3_NC) > 0:
        result_record_whole_NC(args, acc_shot3_NC, shot=3)
    if len(acc_shot5_NC) > 0:
        result_record_whole_NC(args, acc_shot5_NC, shot=5)
    if len(nmi_CL) > 0 and len(ari_CL) > 0:
        result_record_whole_CL(args, nmi_CL, ari_CL)
else:
    print("\n[INFO] Skipping NC and CL result saving for PPI (multi-label)\n")

result_record_whole_LP(args, auc_LP, acc_LP)
print("Done.")
