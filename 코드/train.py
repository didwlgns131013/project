import torch

def train_model(model, vocoder, dataloader, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, (input_data, target_data) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward Pass
            transformed = model(input_data)
            reconstructed = vocoder(transformed)

            # Compute Loss
            loss = loss_fn(reconstructed, target_data)
            total_loss += loss.item()

            # Backward Pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
