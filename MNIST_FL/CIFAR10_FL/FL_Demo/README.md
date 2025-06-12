## Federated Learning ....

In FL using Flower (FL Framework) with the FedAvg (Federated Averaging) algorithm, there are typically two types of model accuracy metrics to consider:

### 1. Aggregated Model Accuracy (AggMoAcc): 
    The accuracy of the global model after aggregation (averaging) of client updates

### 2. Client Model Accuracy (ClMoAcc):
    The accuracy of individual client models before aggregation

## To run the code updated: 
    pip install flwr
    pip install "flwr[simulation]"
## should import all the necessary libaries
    pip install tensorflow
    pip install torch
    pip install torchvison
    pip install numpy
    pip install pandas
## Open different terminal (e.g three terminal): 
    run 
    py server.py, py main.py --cid 1 and py main.py --cid 2


    ##other option to run for MNIST datasets
    ###server (py server.py --num_round =5)
    ###client (py main.py --client_id 0)
    ###client (py main.py --client_id 1)
    ###client (py main.py --client_id 1)
