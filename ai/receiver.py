import socket
from model import Train

# Define the host and port to listen on
HOST = '127.0.0.1'  # localhost
PORT = 12345

# Create a socket object

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Bind the socket to the address and port
    s.bind((HOST, PORT))

    # Listen for incoming connections
    s.listen()
    print("Waiting for connections...")

    # Accept a connection from the Java plugin
    conn, addr = s.accept()
    positions = []
    player_coords = [0, 0, 0]
    while True:
        # Receive data from the Java plugin
        data = conn.recv(100000).decode()
        print('Connected by', addr)

        if "close" in data:
            print('Closing connection...')
            break
        if "Player" in data and "Block" not in data:
            player_coords = data[(data.index(":") + 1):].strip()
            player_coords = [int(player_coords.split(" ")[0]), int(player_coords.split(" ")[1]),
                             int(player_coords.split(" ")[2])]

            print('Player', player_coords)
        else:
            blocks = data.split("%")

            types = [0 if "log" not in block.split("&")[0] else 100 for block in blocks[:-1]]
            positions = [block.split("&")[1][block.split("&")[1].index("x="):-1] for block in blocks[:-1]]
            positions = [position.split(",")[0].split("=")[-1].strip() + " " + position.split(",")[1].split("=")[
                -1].strip() + " " + position.split(",")[2].split("=")[-1].strip() for position in positions]
            positions = [[int(position.split(" ")[0]), int(position.split(" ")[1]), int(position.split(" ")[2])] for
                         position in
                         positions]
            print('Received block types:', types)
            print('Received block positions:', positions)
        if len(positions) > 0:
            combined_data = [[*position, types[i]] for i, position in enumerate(positions)]
            combined_data = [[*player_coords, *data] for data in combined_data]
            print("Combined data: ", combined_data)
            train_model = Train(combined_data)

    conn.close()
