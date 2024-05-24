python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=15 --dirichlet=0.1 --frac=1.0 --num_users=10 --local_ep=1 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=15 --dirichlet=0.3 --frac=1.0 --num_users=10 --local_ep=1 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=15 --dirichlet=0.5 --frac=1.0 --num_users=10 --local_ep=1 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=15 --dirichlet=0.8 --frac=1.0 --num_users=10 --local_ep=1 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=15 --dirichlet=1.0 --frac=1.0 --num_users=10 --local_ep=1 --num_generate=0