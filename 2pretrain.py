from util.data_loader import *
from util.evaluate import *
from util.hyperpara import *
from util.models import *
from util.module import *
from util.training import *
from util.utils import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="cora", help='cora, citeseer, ogbn-arxiv, reddit, Roman-empire')
parser.add_argument('--gpu', type=int, default=0)
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
parser.add_argument('--alpha', type=float, default=1.0)

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

args.result_path = './results_proposed/'
args.eigen_path += f'{args.dataset_name}/'
os.makedirs(args.result_path, exist_ok=True)
os.makedirs(args.model_path, exist_ok=True)
os.makedirs(args.eigen_path, exist_ok=True)

args = SSL_hyperpara(args)
args = SSL_reduction(args)

acc_shot3_NC, acc_shot5_NC, f1_shot3_NC, f1_shot5_NC,auc_LP, acc_LP, nmi_CL, ari_CL = [], [], [], [], [], [], [], []

for i in range(args.nrepeat):
    args.seed += 1
    datasets = get_dataset(args)
    args, data, data_val, data_test = set_dataset(args, datasets)
    print(f"\n[INFO] Iteration {i+1}/{args.nrepeat} | Train nodes: {int(data.train_mask.sum())}")

    num_classes = int(data.y.max().item() + 1) if data.y.dim() == 1 else data.y.size(1)

    train_num = getattr(data, 'train_num_original', int(data.train_mask.sum()))
    args.syn_num = min(
        int(max(train_num * args.reduction_rate, num_classes)),
        train_num
    )

    model_spa = GCN(
        data.num_features,
        args.n_dim,
        num_classes,
        2,
        args.dropout
    ).to(args.device)

    print("\n[Stage 1] Teacher model pretraining...")
    model_spa, cluster_idx = pre_train(args, data, data_test, model_spa)

    save_pre_train_2(args, model_spa, cluster_idx)
    print("[✔] Teacher model saved.\n")

    print("[Stage 2] Evaluating teacher model...\n")
    H, H_val, H_test, H_test_masked, labels_test, \
    H_train_shot_3, label_train_shot_3, \
    H_train_shot_5, label_train_shot_5 = eva_data(args, data, data_val, data_test, model_spa)

    if args.dataset_name != "ppi":
        nmi, ari = evaluate_CL_labels(H_test_masked, labels_test)
        nmi_CL.append(nmi)
        ari_CL.append(ari)
        print(f"[Clustering] NMI: {nmi:.4f}, ARI: {ari:.4f}")

    if args.dataset_name != "ppi":
        acc_5nc, f1_5nc = evaluate_NC(
    args,
    H_train_shot_5,       
    label_train_shot_5,  
    H_test_masked,      
    labels_test,         
    cluster_embs=None,   
    normalize=True
)
        acc_3nc, f1_3nc = evaluate_NC(
    args,
    H_train_shot_3,
    label_train_shot_3,
    H_test_masked,
    labels_test,
    normalize=True
)


        acc_shot3_NC.append(acc_3nc)
        acc_shot5_NC.append(acc_5nc)
        f1_shot3_NC.append(f1_3nc)
        f1_shot5_NC.append(f1_5nc)
        print(f"[Node Classification] Accuracy 3-shot: {acc_3nc:.4f}, Accuracy 5-shot: {acc_5nc:.4f}")
        print(f"[Node Classification] F1 3-shot: {f1_3nc:.4f}, F1 5-shot: {f1_5nc:.4f}")

    auc_lp, acc_lp = evaluate_LP(args, data, H, H_val, H_test, data_val, data_test)
    auc_LP.append(auc_lp)
    acc_LP.append(acc_lp)
    print(f"[Link Prediction] AUC: {auc_lp:.4f}, ACC: {acc_lp:.4f}")

print("\n[Stage 3] Saving pretraining results...")
pretrain_record_caption(args)

if args.dataset_name != "ppi":
    result_record_whole_NC(args, acc_shot3_NC, shot=3)
    result_record_whole_NC(args, acc_shot5_NC, shot=5)
    result_record_whole_CL(args, nmi_CL, ari_CL)
else:
    print("[INFO] Skipping NC/CL metrics for PPI dataset.")

result_record_whole_LP(args, auc_LP, acc_LP)
print("\n✅ Pretraining complete.\n")
