import threading
import time
from client import start_client
from server import start_server

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    time.sleep(5)  # Wait for the server to start

    hospital_data_dirs = [
        "D:/Colposcopy/Project2024/hospital1/data",
        "D:/Colposcopy/Project2024/hospital2/data",
        "D:/Colposcopy/Project2024/hospital3/data"
    ]

    client_threads = []
    for data_dir in hospital_data_dirs:
        client_thread = threading.Thread(target=start_client, args=(data_dir,))
        client_threads.append(client_thread)
        
    for client_thread in client_threads:
        client_thread.start()

    server_thread.join()
    for client_thread in client_threads:
        client_thread.join()
