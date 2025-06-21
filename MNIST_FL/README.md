To run the code of MNIST
py server.py --num_rounds 3
py main.py --client_id 0
py main.py --client_id 1
py main.py --client_id 2


py server.py --num_rounds 10  
python client.py --cid 1 
cifar_main.py --client_id 2 --num_clients 3 

git clone --depth=1 https://github.com/adap/flower.git _tmp; Move-Item _tmp/examples/quickstart-huggingface .; Remove-Item -Recurse -Force _tmp; Set-Location quickstart-huggingface

python main.py --num_clients 2 --csv/labels.csv --img_dir path/to/images --num_classes 4 (for Colposcopy data)
