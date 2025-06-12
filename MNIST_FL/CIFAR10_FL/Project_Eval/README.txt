Checking the evaluation of the Cifar10 is run as follows
python main.py --role server --num_clients 3 --epochs 5

python main.py --role client --cid 0 --num_clients 3 --batch_size 32
python main.py --role client --cid 1 --num_clients 3 --batch_size 32
python main.py --role client --cid 2 --num_clients 3 --batch_size 32
