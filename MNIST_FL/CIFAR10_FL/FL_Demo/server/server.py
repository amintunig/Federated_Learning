import flwr as fl

def start_server():
    fl.server.start_server(server_address="localhost:8080", config={"num_rounds": 3})

if __name__ == "__main__":
    start_server()
