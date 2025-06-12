import threading
from client import start_client
from server import start_server

if __name__ == "__main__":
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    # Start clients for three different hospitals in separate threads
    hospital_data_dirs = [
        "path/to/hospital1/data",
        "path/to/hospital2/data",
        "path/to/hospital3/data"
    ]

    client_threads = []
    
    for data_dir in hospital_data_dirs:
        client_thread = threading.Thread(target=start_client, args=(data_dir,))
        client_threads.append(client_thread)
        
    for client_thread in client_threads:
        client_thread.start()

    # Wait for all threads to complete
    server_thread.join()