from util.data_loader import *
from util.evaluate import *
from util.hyperpara import *
from util.models import *
from util.module import *
from util.training import *
from util.utils import *
import torch
import torch.nn.functional as F
import warnings
import numpy as np
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="Roman-empire", help='cora, citeseer, ogbn-arxiv, reddit')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--result_path', type=str, default="./results/")
parser.add_argument('--model_path', type=str, default="./save_pretrain_model/")
parser.add_argument('--eigen_path', type=str, default="./save_eigen/")
parser.add_argument('--condensed_path', type=str, default="./save_condensed_data/")
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--split_data_dir', type=str, default="./dataset_split/")

parser.add_argument('--epoch_u', type=int, default=600)
parser.add_argument('--lr_u', type=float, default=0.001)
parser.add_argument('--epoch_x', type=int, default=600)
parser.add_argument('--lr_x', type=float, default=0.001)
parser.add_argument('--epoch_downstream', type=int, default=600)
parser.add_argument('--lr_downstream', type=float, default=0.01)
parser.add_argument('--loss_generation', type=str, default='mse')

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
args.condensed_path += f'{args.dataset_name}/'
args.eigen_path += f'{args.dataset_name}/'
os.makedirs(args.result_path, exist_ok=True)
os.makedirs(args.condensed_path, exist_ok=True)
os.makedirs(args.eigen_path, exist_ok=True)

args = generation_hyperpara(args)

def fewshot_prototype_eval(H_train_shot, y_train_shot, H_test, y_test, device):
    H_train_shot = H_train_shot.to(device)
    H_test = H_test.to(device)
    y_train_shot = y_train_shot.to(device)
    y_test = y_test.to(device)

    d = H_train_shot.size(1)
    classes = torch.unique(y_train_shot)
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
    H_test = H_test.to(device)
    cluster_centroids = cluster_centroids.to(device)
    H_test_norm = F.normalize(H_test, p=2, dim=-1)
    cluster_norm = F.normalize(cluster_centroids, p=2, dim=-1)
    sims = torch.matmul(H_test_norm, cluster_norm.T)
    preds = torch.argmax(sims, dim=1)
    acc = (preds == y_test.to(device)).float().mean().item()
    return acc

acc_nc_list = []
auc_LP = []
acc_LP = []
nmi_CL = []
ari_CL = []

for i in range(args.nrepeat):
    args.seed += 1

    datasets = get_dataset(args)
    args, data, data_val, data_test = set_dataset(args, datasets)

    train_num = int(getattr(data, 'train_num_original', int(data.train_mask.sum())))
    num_classes = int(data.y.max().item() + 1) if data.y.dim() == 1 else data.y.size(1)
    requested = int(train_num * args.reduction_rate)
    args.syn_num = int(min(max(requested, num_classes), max(1, train_num)))
    print(f"syn_num: {args.syn_num} (requested {requested}, classes {num_classes}, train {train_num})")

    data = load_eigens(args, data)

    model_spa = GCN(data.num_features, args.n_dim, args.syn_num, 2, args.dropout).to(args.device)
    model_spe = EigenMLP(args.n_dim, args.syn_num, args.n_dim).to(args.device)
    model_spa, ccenter_spa, model_spe, ccenter_spe = load_pre_train(args, model_spa, model_spe)

    if isinstance(ccenter_spa, np.ndarray):
        ccenter_spa = torch.from_numpy(ccenter_spa).float()
    else:
        ccenter_spa = ccenter_spa.detach().float()

    if isinstance(ccenter_spe, np.ndarray):
        ccenter_spe = torch.from_numpy(ccenter_spe).float()
    else:
        ccenter_spe = ccenter_spe.detach().float()

    adj_syn = adj_generation(args, data, model_spe, ccenter_spe)
    data_syn = feat_generation(args, data, model_spa, ccenter_spa, adj_syn)
    save_condensed_data(args, data_syn)

    model = train_model_syn(args, data_syn)

    H, H_val, H_test, H_test_masked, labels_test, \
    H_train_shot_3, label_train_shot_3, \
    H_train_shot_5, label_train_shot_5 = eva_data(args, data, data_val, data_test, model)

    cluster_label_embs = F.normalize(ccenter_spa.to(args.device), dim=1)

    acc_3 = None
    acc_5 = None
    acc_centroid = None

    if not (args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1)):
        if isinstance(H_train_shot_3, np.ndarray):
            H_train_shot_3 = torch.from_numpy(H_train_shot_3).float()
        if isinstance(H_train_shot_5, np.ndarray):
            H_train_shot_5 = torch.from_numpy(H_train_shot_5).float()
        if isinstance(label_train_shot_3, np.ndarray):
            label_train_shot_3 = torch.from_numpy(label_train_shot_3).long()
        if isinstance(label_train_shot_5, np.ndarray):
            label_train_shot_5 = torch.from_numpy(label_train_shot_5).long()

        acc_3 = fewshot_prototype_eval(H_train_shot_3, label_train_shot_3, H_test_masked, labels_test, args.device)
        acc_5 = fewshot_prototype_eval(H_train_shot_5, label_train_shot_5, H_test_masked, labels_test, args.device)

        acc_centroid = cluster_centroid_eval(H_test_masked, labels_test, ccenter_spa, args.device)

    print(f"[Eval] Few-shot3: {acc_3}, Few-shot5: {acc_5}, Centroid acc: {acc_centroid}")

    if args.dataset_name == "ppi" or (data.y.dim() == 2 and data.y.size(1) > 1):
        nmi, ari = 0.0, 0.0
    else:
        nmi, ari = evaluate_CL_labels(H_test_masked, labels_test)

    auc_lp_run, acc_lp_run = evaluate_LP(args, data, H, H_val, H_test, data_val, data_test)

    acc_nc_list.append(acc_centroid if acc_centroid is not None else 0.0)
    auc_LP.append(auc_lp_run)
    acc_LP.append(acc_lp_run)
    nmi_CL.append(nmi)
    ari_CL.append(ari)

    print(f"[Run {i+1}] Centroid NC acc: {100*(acc_centroid if acc_centroid else 0):.2f} | AUC LP: {100*auc_lp_run:.2f} | NMI: {100*nmi:.2f}")

downstream_record_caption(args, data_syn)
if len(acc_nc_list) > 0:
    result_record_whole_NC(args, acc_nc_list, shot=3)
    result_record_whole_NC(args, acc_nc_list, shot=5)
result_record_whole_LP(args, auc_LP, acc_LP)
result_record_whole_CL(args, nmi_CL, ari_CL)

print("\n✅ Finished Condensation Evaluation with Embedding-based Label Semantics.\n")
