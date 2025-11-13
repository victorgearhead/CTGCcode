from util.data_loader import *
from util.evaluate import *
from util.hyperpara import *
from util.models import *
from util.module import *
from util.training import *
from util.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="cora", help= 'cora, citeseer, ogbn-arxiv, reddit')
parser.add_argument('--result_path', type=str, default="./results/")
parser.add_argument('--eigen_path', type=str, default="./save_eigen/")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--shot', type=int, default=3)
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--split_data_dir', type=str, default="./dataset_split/")
args = parser.parse_args()

print(f"dataset:{args.dataset_name}")
args = device_setting(args)
seed_everything(args.seed)

args.eigen_path +=  f'{args.dataset_name}/'
if not os.path.exists(args.eigen_path):
    os.makedirs(args.eigen_path)
args.eigvals_path = args.eigen_path+ "eigenvalues.npy"
args.eigvecs_path = args.eigen_path+ "eigenvectors.npy"
args.lccmask_path = args.eigen_path+ "mask_lcc.npy"

datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)

L_lcc = aug_full_connected(data.x, data.edge_index, data.num_nodes)
eigenvals_lcc, eigenvecs_lcc = get_eigens(args, L_lcc)

np.save(args.eigvals_path, eigenvals_lcc)
np.save(args.eigvecs_path, eigenvecs_lcc)