import os, torch, warnings, argparse
import torch.nn.functional as F
from util.data_loader import *
from util.evaluate import *
from util.hyperpara import *
from util.models import *
from util.module import *
from util.training import *
from util.utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="Roman-empire", help='cora, citeseer, ogbn-arxiv, reddit')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--shot', type=int, default=3)
parser.add_argument('--result_path', type=str, default="./results/")
parser.add_argument('--model_path', type=str, default="./save_pretrain_model/")
parser.add_argument('--eigen_path', type=str, default="./save_eigen/")
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--split_data_dir', type=str, default="./dataset_split/")
parser.add_argument('--epoch_cls', type=int, default=200)
parser.add_argument('--epoch_lp', type=int, default=200)
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--lr_lp', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--nrepeat', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_dim', type=int, default=256)
parser.add_argument('--test_gnn', type=str, default='GCN')
args = parser.parse_args()

args = device_setting(args)
seed_everything(args.seed)

args.result_path = './results_whole/'
os.makedirs(args.result_path, exist_ok=True)
os.makedirs(args.split_data_dir, exist_ok=True)

acc_shot3_NC, acc_shot5_NC = [], []
auc_LP, acc_LP, nmi_CL, ari_CL = [], [], [], []

for i in range(args.nrepeat):
    args.seed += i
    seed_everything(args.seed)
    datasets = get_dataset(args)
    args, data, data_val, data_test = set_dataset(args, datasets)
    print(f"train num: {int(data.train_mask.sum())}")
    num_classes = int(data.y.max().item() + 1) if data.y.dim() == 1 else data.y.size(1)
    model = GCN(data.num_features, args.n_dim, num_classes, 2, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.weight_decay)
    criterion = torch.nn.NLLLoss()

    best_val_acc, best_loss = 0, 1e6
    for epoch in range(1, args.epoch_cls + 1):
        if epoch == args.epoch_cls // 2:
            lr = args.lr_cls * 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.train()
        output = model(data)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.dataset_name in ['flickr', 'reddit']:
            train_acc, val_acc, tmp_test_acc = test_inductive(args, model, data_val, data_test)
        else:
            train_acc, val_acc, tmp_test_acc = test(model, data)

        if val_acc > best_val_acc:
            best_val_acc, test_acc, best_weights = val_acc, tmp_test_acc, model.state_dict()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] Train acc: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}")

    model.load_state_dict(best_weights)

    H, H_val, H_test, H_test_masked, labels_test, \
    H_train_shot_3, label_train_shot_3, \
    H_train_shot_5, label_train_shot_5 = eva_data(args, data, data_val, data_test, model)

    nmi, ari = evaluate_CL_labels(H_test_masked, labels_test)
    auc_lp, acc_lp = evaluate_LP(args, data, H, H_val, H_test, data_val, data_test)

    if args.dataset_name not in ["ppi"]:
        acc3, f13 = evaluate_NC(args, H_train_shot_3, label_train_shot_3,
                                 H_test_masked, labels_test)
        
        acc5, f15 = evaluate_NC(args, H_train_shot_5, label_train_shot_5,
                                 H_test_masked, labels_test)
        acc_shot3_NC.append(acc3)
        acc_shot5_NC.append(acc5)
    else:
        print("[INFO] Skipping NC eval for PPI (multi-label).")

    auc_LP.append(auc_lp)
    acc_LP.append(acc_lp)
    nmi_CL.append(nmi)
    ari_CL.append(ari)
    print()

result_record_whole_NC(args, acc_shot3_NC, shot=3)
result_record_whole_NC(args, acc_shot5_NC, shot=5)
result_record_whole_LP(args, auc_LP, acc_LP)
result_record_whole_CL(args, nmi_CL, ari_CL)
print("\n✅ Finished full-graph baseline training & evaluation.\n")




