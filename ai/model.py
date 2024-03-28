import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import formatting


class BlockModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BlockModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define the input size, hidden size, and output size of the model
input_size = 4  # x, y, z coordinates, and block type
hidden_size = 64
output_size = 4  # Number of keys to move (e.g., up, down, left, right)

# Instantiate the model
model = BlockModel(input_size, hidden_size, output_size)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example input: nearest blocks represented as a numpy array
nearest_blocks = np.array([[1, 2, 3, 0], [4, 5, 6, 1], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0], [7, 8, 9, 0]])

# Convert input to PyTorch tensor and move to GPU
inputs = torch.tensor(nearest_blocks, dtype=torch.float32).to(device)

# Forward pass
outputs = model(inputs)

# Apply softmax to convert outputs to probabilities
probabilities = torch.softmax(outputs, dim=1)

# Output will be probabilities for each key
# You can use argmax to get the index of the highest probability
predicted_key = torch.argmax(probabilities, dim=1)

print("Predicted key probabilities:", probabilities.tolist())
movement_mapping = {
    0: "w",
    1: "a",
    2: "s",
    3: "d"
}

# Convert predicted keys to WASD movements
predicted_movements = [movement_mapping[key] for key in predicted_key.tolist()]

print("Predicted movements:", predicted_movements)
time.sleep(10)
game = formatting.Game(predicted_movements)
