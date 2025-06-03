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
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, r2_score


def train_model(model, backend, preprocessessor, dataloader_opts, optimizer, dataset_path, criterion, device, epochs):
    
    dataset = SceneDataset(dataset_path, dataloader_opts)
    if dataloader_opts["inorder"] == "Always":
        dataset.set_inorder(True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=None)
    model.to(device)
    backend.to(device)
    writer = SummaryWriter(log_dir="./runs/")
    save_path = Path("./checkpoints/")
    save_path.mkdir(parents=True, exist_ok=True)

    ts_preds, ts_labels = [], []
    to_preds, to_labels = [], []
    vr_preds, vr_labels = [], []
    vt_preds, vt_labels = [], []

    for epoch_iter in tqdm(range(epochs*len(dataloader))):
        epoch = epoch_iter // len(dataloader)
        cur_iter = epoch_iter % len(dataloader)

        model.train()
        
        scene = dataset[cur_iter]

        output = model(scene)
        label = scene[0].get_label()

        label = {key : torch.tensor(float(label[key]), dtype=torch.float32).to(device) for key in label.keys()}

        ts_preds.append(output["ts"].item())
        ts_labels.append(label["Ts"].item())

        to_preds.append(output["to_output"].item())
        to_labels.append(label["To"].item())

        vr_preds.append(output["vr"].item())
        vr_labels.append(label["Vr"].item())

        vt_preds.append(output["vt"].item())
        vt_labels.append(label["Vt"].item())
        
        print(f"ts: {label['Ts']} : {output['ts']}")
        print(f"to: {label['To']} : {output['to_output']}")
        print(f"vr: {label['Vr']} : {output['vr']}")
        print(f"vt: {label['Vt']} : {output['vt']}")
        print(f"ts: {label['Ts'].item()} : {output['ts'].item()}")
        print(f"to: {label['To'].item()} : {output['to_output'].item()}")
        print(f"vr: {label['Vr'].item()} : {output['vr'].item()}")
        print(f"vt: {label['Vt'].item()} : {output['vt'].item()}")

        optimizer.zero_grad()

        loss_ts = criterion(output["ts"], label["Ts"])
        mask = loss_ts.item() == 1

        loss_to = criterion(output["to_output"], label["To"]) * mask
        loss_vr = criterion(output["vr"], label["Vr"]) * mask
        loss_vt = criterion(output["vt"], label["Vt"]) * mask

        loss = loss_ts + loss_to + loss_vr + loss_vt

        loss.backward()
        optimizer.step()

        # log loss
        print(f"Epoch {epoch + 1}/{epochs}, Iteration {cur_iter + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        print(f"Ts Loss: {loss_ts.item():.4f}")
        print(f"To Loss: {loss_to.item():.4f}")
        print(f"Vr Loss: {loss_vr.item():.4f}")
        print(f"Vt Loss: {loss_vt.item():.4f}")

        writer.add_scalar("Loss/Total", loss.item(), epoch_iter)
        writer.add_scalar("Loss/ts", loss_ts.item(), epoch_iter)
        writer.add_scalar("Loss/to", loss_to.item(), epoch_iter)
        writer.add_scalar("Loss/vr", loss_vr.item(), epoch_iter)
        writer.add_scalar("Loss/vt", loss_vt.item(), epoch_iter)
        
        if cur_iter == len(dataloader) - 1:
            tqdm.write(f"Epoch {epoch + 1}/{epochs} completed.")

        
        ts_mae, ts_r2 = generate_metrics(ts_preds, ts_labels)
        to_mae, to_r2 = generate_metrics(to_preds, to_labels)
        vr_mae, vr_r2 = generate_metrics(vr_preds, vr_labels)
        vt_mae, vt_r2 = generate_metrics(vt_preds, vt_labels)

        print(f"Epoch {epoch + 1} Evaluation Metrics:")
        print(f"Ts MAE: {ts_mae:.4f}, R2: {ts_r2:.4f}")
        print(f"To MAE: {to_mae:.4f}, R2: {to_r2:.4f}")
        print(f"Vr MAE: {vr_mae:.4f}, R2: {vr_r2:.4f}")
        print(f"Vt MAE: {vt_mae:.4f}, R2: {vt_r2:.4f}")

        writer.add_scalars("Eval/MAE", {
        "Ts": ts_mae,
        "To": to_mae,
        "Vr": vr_mae,
        "Vt": vt_mae
        }, epoch)

        writer.add_scalars("Eval/R2", {
        "Ts": ts_r2,
        "To": to_r2,
        "Vr": vr_r2,
        "Vt": vt_r2
        }, epoch)
        
        ts_preds, ts_labels = [], []
        to_preds, to_labels = [], []
        vr_preds, vr_labels = [], []
        vt_preds, vt_labels = [], []
        torch.save(model.state_dict(), save_path / f"epoch_{epoch + 1}.pt") #checkpoint

    writer.close()

def generate_metrics(preds, targets):
    preds_np = np.array(preds)
    targets_np = np.array(targets)
    mae = mean_absolute_error(targets_np, preds_np)
    r2 = r2_score(targets_np, preds_np)
    return mae, r2

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
