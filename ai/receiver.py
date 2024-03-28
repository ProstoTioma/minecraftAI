import socket

# Define the host and port to listen on
HOST = '127.0.0.1'  # localhost
PORT = 12345

# Create a socket object

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Bind the socket to the address and port
    s.bind((HOST, PORT))

    # Listen for incoming connections
    s.listen()

    # Accept a connection from the Java plugin
    conn, addr = s.accept()
    while True:
        print('Connected by', addr)

        # Receive data from the Java plugin
        data = conn.recv(1024).decode()
        print('Received data:', data)
