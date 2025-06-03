from poolNet import PoolNetHead, LSTMHead, ViTModel, ViTImageProcessor
# NOTE: VitImageProcessor does not crop. non-square images will be squished vertically or horizontally
import torch
from PIL import Image
import numpy as np
import tensorboard
import matplotlib.pyplot as plt
from pathlib import Path
from dataloader import SceneDataset, collate_fn
from tqdm import tqdm

def train_model(model, backend, preprocessessor, dataloader_opts, optimizer, dataset_path, criterion, device, epochs):
    
    dataset = SceneDataset(dataset_path, dataloader_opts)
    if dataloader_opts["inorder"] == "Always":
        dataset.set_inorder(True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=None)
    model.to(device)
    backend.to(device)

    for epoch_iter in tqdm(range(epochs*len(dataloader))):
        epoch = epoch_iter // len(dataloader)
        cur_iter = epoch_iter % len(dataloader)

        model.train()
        
        scene = dataset[cur_iter]

        output = model(scene)
        label = scene.get_label()
        label = torch.tensor(label, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # log loss
        print(f"Epoch {epoch + 1}/{epochs}, Iteration {cur_iter + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        if cur_iter == len(dataloader) - 1:
            tqdm.write(f"Epoch {epoch + 1}/{epochs} completed.")

def main():
    backend = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMHead(input_dim=768, hidden_dim=256, num_layers=2, device=device)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    epochs = 2

    dataloader_opts = {
        "augment": True,
        "batch_size": 1,
        "shuffle": True,
        "ShiftAug" : "Static",
        "ColorShiftAug" : True,
        "ResizeAug" : "Static",
        "inorder" : "Always"
    }

    dataset_path = "/media/SharedStorage/redwood/output"

    image_dir = Path("test-videos")
    image_path = image_dir / "non-square.jpg"
    image = Image.open(image_path).convert("RGB")
    image = processor(images=image, return_tensors="pt")
    image = image["pixel_values"]
    image_np = image.numpy()

    # plt.imshow(image_np[0].transpose(1, 2, 0))  # Convert to HWC format for plotting
    # plt.show()

    train_model(model, backend, processor, dataloader_opts, optimizer, dataset_path, criterion, device, epochs)

if __name__ == "__main__":
    main()
