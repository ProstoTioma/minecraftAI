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


class Train:
    def __init__(self, inputs):
        input_size = 7  # x, y, z coordinates, and block type, and player coordinates
        hidden_size = 64
        output_size = 4  # Number of keys to move (e.g., up, down, left, right)
        self.num_epochs = 10

        self.model = BlockModel(input_size, hidden_size, output_size)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.train_model(inputs)

    def train_model(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        # Assuming you have labels for training, otherwise, you need to provide them
        labels = torch.tensor([0] * len(inputs)).to(self.device)  # Update this with actual labels

        for epoch in range(self.num_epochs):  # You need to define num_epochs
            # Forward pass
            outputs = self.model(inputs)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {loss.item()}")

        # After training, you can proceed with inference
        self.inference(inputs)

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs)

            # Apply softmax to convert outputs to probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # Output will be probabilities for each key
            # You can use argmax to get the index of the highest probability
            predicted_key = torch.argmax(probabilities, dim=1)

            movement_mapping = {
                0: "w",
                1: "a",
                2: "s",
                3: "d"
            }

            print("Predicted key probabilities:", probabilities.tolist())

            # Convert predicted keys to WASD movements
            predicted_movements = [movement_mapping[key] for key in predicted_key.tolist()]

            print("Predicted movements:", predicted_movements)
            game = formatting.Game(predicted_movements)
