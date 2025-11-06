device=0

data=cora
for r in 3 5
do
python3 whole.py --gpu $device --dataset $data --shot $r
done
python3 whole_LP.py --gpu $device --dataset $data

data=citeseer
for r in 3 5
do
python3 whole.py --gpu $device --dataset $data --shot $r
done
python3 whole_LP.py --gpu $device --dataset $data

data=reddit
for r in 3 5
do
python3 whole.py --gpu $device --dataset $data --shot $r
done
python3 whole_LP.py --gpu $device --dataset $data

data=ogbn-arxiv
for r in 3 5
do
python3 whole.py --gpu $device --dataset $data --shot $r
done
python3 whole_LP.py --gpu $device --dataset $data