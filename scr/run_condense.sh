device=0

for r in 0.25 0.5 1
do
python3 4condense.py --gpu $device --dataset cora --reduction_rate $r
done

for r in 0.25 0.5 1
do
python3 4condense.py --gpu $device --dataset citeseer --reduction_rate $r
done

for r in 0.001 0.005 0.01
do
python3 4condense.py --gpu $device --dataset ogbn-arxiv --reduction_rate $r 
done

for r in 0.0005 0.001 0.002
do
python3 4condense.py --gpu $device --dataset reddit --reduction_rate $r 
done