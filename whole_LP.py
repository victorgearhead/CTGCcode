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
parser.add_argument('--dataset_name', type=str, default="Roman-empire")
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--result_path', type=str, default="./results/")
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--split_data_dir', type=str, default="./dataset_split/")
parser.add_argument('--epoch_lp', type=int, default=200)
parser.add_argument('--lr_lp', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--nrepeat', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_dim', type=int, default=256)
parser.add_argument('--test_gnn', type=str, default='GCN')
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--epoch_cls', type=int, default=200)
parser.add_argument('--shot', type=int, default=3)
args = parser.parse_args()

args = device_setting(args)
seed_everything(args.seed)

args.result_path = './results_whole/'
os.makedirs(args.result_path, exist_ok=True)
os.makedirs(args.split_data_dir, exist_ok=True)

datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)

auc_all, acc_all = [], []

for i in range(args.nrepeat):
    args.seed += i
    seed_everything(args.seed)
    num_classes = int(data.y.max().item() + 1) if data.y.dim() == 1 else data.y.size(1)
    model = GCN(data.num_features, args.n_dim, num_classes, 2, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_lp, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    neg_edge_index = train_negative_sampling(data)

    best_val_acc = best_test_acc = best_test_auc = 0
    for epoch in range(1, args.epoch_lp + 1):
        if epoch == args.epoch_lp // 2:
            lr = args.lr_lp * 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.train()
        logits = model.encode(data.x, data.edge_index)
        edge_label_index, edge_label = edge_negative_sampling(data, neg_edge_index)
        out = model.decode(logits, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        train_auc = roc_auc_score(edge_label.cpu().numpy(),
                                  out.view(-1).sigmoid().detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.dataset_name in ['flickr', 'reddit']:
            _, val_auc, tmp_test_auc, val_acc, tmp_test_acc = test_LP_inductive(args, model, data_val, data_test)
        else:
            _, val_auc, tmp_test_auc, val_acc, tmp_test_acc = test_LP(model, data)

        if val_acc > best_val_acc:
            best_val_acc, best_test_acc, best_test_auc = val_acc, tmp_test_acc, tmp_test_auc

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, "
                  f"Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, Test AUC: {tmp_test_auc:.4f}")

    auc_all.append(best_test_auc)
    acc_all.append(best_test_acc)
    print()

result_record_whole_LP(args, auc_all, acc_all)
print("\n✅ Finished full-graph link prediction baseline.\n")
