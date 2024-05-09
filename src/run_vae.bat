python federated_main.py --model=vae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=30 --dirichlet=0.2 --frac=1.0 --num_users=10 --local_ep=10
python federated_main.py --model=vae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=30 --dirichlet=0.4 --frac=1.0 --num_users=10 --local_ep=10
python federated_main.py --model=vae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=30 --dirichlet=0.6 --frac=1.0 --num_users=10 --local_ep=10
python federated_main.py --model=vae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=30 --dirichlet=0.8 --frac=1.0 --num_users=10 --local_ep=10
python federated_main.py --model=vae --dataset=mnist --gpu=cuda:0 --iid=2 --epochs=30 --dirichlet=1.0 --frac=1.0 --num_users=10 --local_ep=10

