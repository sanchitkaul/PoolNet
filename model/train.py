from poolNet import PoolNetHead, LSTMHead, ViTModel, ViTImageProcessor, BasicCNN, SequentialTransformer, ConvFormer
# NOTE: VitImageProcessor does not crop. non-square images will be squished vertically or horizontally
import torch
from PIL import Image
import numpy as np
import tensorboard
import matplotlib.pyplot as plt
from pathlib import Path
from dataloader import SceneDataset, SequentialFrameDataset, collate_fn, FrameMatchingDataset, FrameMatchingDatasetNoGeometry, FrameSequenceMeshification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, r2_score
import json
import math

def test_model(model, test_dataset):

    ts_total = 0
    to_total = 0
    vr_total = 0
    vt_total = 0

    for i in range(len(test_dataset)):
        scene = test_dataset[i]
        model.eval()
        with torch.no_grad():
            output = model(scene)
            label = scene[0].get_label()
            print(f"Scene {i}:")
            print(f"Ts: {label['Ts']}, Predicted: {output['ts'].item()}")
            print(f"To: {label['To']}, Predicted: {output['to_output'].item()}")
            print(f"Vr: {label['Vr']}, Predicted: {output['vr'].item()}")
            print(f"Vt: {label['Vt']}, Predicted: {output['vt'].item()}")

            MAE_ts = mean_absolute_error([label['Ts']], [output['ts'].item()])
            MAE_to = mean_absolute_error([label['To']], [output['to_output'].item()])
            MAE_vr = mean_absolute_error([label['Vr']], [output['vr'].item()])
            MAE_vt = mean_absolute_error([label['Vt']], [output['vt'].item()])

            ts_total = MAE_ts
            to_total = MAE_to
            vr_total = MAE_vr
            vt_total = MAE_vt

    return ts_total / len(test_dataset), to_total / len(test_dataset), vr_total / len(test_dataset), vt_total / len(test_dataset)

def train_model(model, backend, preprocessessor, dataloader_opts, optimizer, dataset_path, criterion, device, epochs):
    
    dataset = SceneDataset(dataset_path, dataloader_opts)
    test_size = 0.2
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size
    train_dataset, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    negatives = []
    for i in range(len(dataset)):
        scene = dataset[i][0]
        if scene.get_label()["Ts"] == 0:
            negatives.append(i)
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

    prev_positive = False
    in_pretraining = False

    test_results = [test_model(model, test_data)]

    for epoch_iter in tqdm(range(epochs*len(train_dataset))):
        epoch = epoch_iter // len(train_dataset)
        cur_iter = epoch_iter % len(train_dataset)

        if in_pretraining and prev_positive == True and cur_iter not in negatives:
            continue
        else:
            model.train()
            
            scene = train_dataset[cur_iter]

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
            
            print(f"ts: {label['Ts'].item()} : {output['ts'].item()}")
            print(f"to: {label['To'].item()} : {output['to_output'].item()}")
            print(f"vr: {label['Vr'].item()} : {output['vr'].item()}")
            print(f"vt: {label['Vt'].item()} : {output['vt'].item()}")

            optimizer.zero_grad()

            loss_ts = criterion(output["ts"], label["Ts"])
            print(f"Loss Ts: {loss_ts.item()}")
            print(f"Loss {loss_ts.item() > 0.5}")
            mask = loss_ts.item() > 0.5

            prev_positive = mask

            loss_to = criterion(output["to_output"], label["To"])
            loss_vr = criterion(output["vr"], label["Vr"])
            loss_vt = criterion(output["vt"], label["Vt"])

            # loss = loss_ts + (loss_to + loss_vr + loss_vt*0.1)/2
            loss = loss_vr + (loss_vt * 0.1)

            loss.backward()
            optimizer.step()

            # log loss
            print(f"Epoch {epoch + 1}/{epochs}, Iteration {cur_iter + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            print(f"To Loss: {loss_to.item():.8f}")
            print(f"Vr Loss: {loss_vr.item():.8f}")
            print(f"Ts Loss: {loss_ts.item():.8f}")
            print(f"Vt Loss: {loss_vt.item():.8f}")

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

            scene[0].reset()

            torch.save(model.state_dict(), save_path / f"epoch_{epoch + 1}.pt") #checkpoint
            if cur_iter % len(train_dataset) == len(train_dataset) - 1:
                test_results.append(test_model(model, test_data))

        if in_pretraining and epoch == 10:
            in_pretraining = False

    print(test_results)
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
    plt.plot([i for i in range(len(test_results))], [res[0] for res in test_results], label='Ts MAE')
    plt.savefig("ts_mae.png")
    plt.plot([i for i in range(len(test_results))], [res[1] for res in test_results], label='To MAE')
    plt.savefig("to_mae.png")
    plt.plot([i for i in range(len(test_results))], [res[2] for res in test_results], label='Vr MAE')
    plt.savefig("vr_mae.png")
    plt.plot([i for i in range(len(test_results))], [res[3] for res in test_results], label='Vt MAE')
    plt.savefig("vt_mae.png")

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

    model = LSTMHead(input_dim=2048, hidden_dim=256, num_layers=2, device=device)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = torch.nn.L1Loss()

    epochs = 10

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

def percent_error_loss(outputs, labels):
    # Avoid division by zero
    epsilon = 1e-8
    return torch.mean(torch.abs((outputs - labels) / (labels + epsilon)))

def main2():
    

    # writer = SummaryWriter(log_dir="./basic_runs/")

    # good loss function for single floating point value
    torch.manual_seed(48)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion = torch.nn.HuberLoss()
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    # model = BasicCNN().to(device)
    # model = LSTMHead().to(device)
    model = SequentialTransformer(2).to(device)
    # optim = torch.optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=0.0001)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    # optim = torch.optim.rmsprop(model.parameters(), lr=0.0001, weight_decay=0.0001)
    
    epochs = 100

    sequential_dataset = SequentialFrameDataset(Path("/media/SharedStorage/redwood/output"))

    train_loader = torch.utils.data.DataLoader(
        sequential_dataset,
        batch_size=32,
        shuffle=True,
    )
    
    total_loss = 0.0
    loss_count = 0
    reg_threshold = 3
    for epoch in range(epochs):
        model.train()
        for i, batch in tqdm(enumerate(train_loader)):
            # print("batch")
            # print(type(batch))
            # print(len(batch))
            # print(batch)
            # print("done")
            images, labels = batch
            images = images.to(device).float()
            labels = labels.to(device).float()

            optim.zero_grad()
            outputs = model(images).float()
            # loss = percent_error_loss(outputs, labels)
            loss = criterion(outputs, labels).mean().float()
            

            epsilon = 1e-8
            abs_error = [abs(((outputs[j] - labels[j])).detach().cpu().numpy()) for j in range(len(outputs))]
            # if np.mean(percent_error) == np.inf:
                # print(f"inf encountered, came from {outputs} - {labels}")
            output_range = outputs.max() - outputs.min()

            
            pre_reg = loss.clone().item()
            penalty = (reg_threshold * 0.1) / (output_range + epsilon)
            # if output_range < reg_threshold:
            loss = loss + ((reg_threshold* 0.1)/(output_range + epsilon))
            loss_count += 1

            total_loss += loss.item()
            loss.backward()
            optim.step()
            if i % 50 == 0:            
                print()
                print("-----------------------------------------------------------")
                print("output: ", end="")
                for j in range(len(outputs)):
                    print(f"{outputs[j].item():.3f}", end=",")
                print()
                print("labels: ", end="")
                for j in range(len(labels)):
                    print(f"{labels[j].item():.3f}", end=",")
                print()
                print("prcerr: ", end="")
                for j in range(len(abs_error)):
                    val = float(abs_error[j])
                    print(f"{val:.3f}", end=",")
                print()
                print(f"range: {output_range.item():.5f}, penalty: {penalty} pre-reg: {pre_reg:.5f}, loss: {loss.item():.5f}")
                
                print(f"Epoch {epoch + 1}/{epochs}, iter {i + 1 // len(sequential_dataset) / train_loader.batch_size}, loss: {total_loss / loss_count:.4f}")
                print("-----------------------------------------------------------")
                total_loss = 0.0
                loss_count = 0
            


        # torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

import matplotlib.pyplot as plt
import torch.nn.functional as F

def main3():
    torch.manual_seed(51)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicCNN(320, 240, 3, num_classes=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    epochs = 10
    frame_match_only_dataset = FrameMatchingDatasetNoGeometry(Path("/media/SharedStorage/redwood/FramesOnly"))
    test_ratio = 0.1
    test_size = int(len(frame_match_only_dataset) * test_ratio)
    train_size = len(frame_match_only_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(frame_match_only_dataset, [train_size, test_size])
    # frame_match_dataset = FrameMatchingDataset(Path("/media/SharedStorage/redwood/output"))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    total_loss = 0.0
    loss_count = 0

    criterion = torch.nn.CosineEmbeddingLoss(margin=0.1)

    for epoch in range(epochs):

        model.eval()
        test_correct = 0
        test_count = 0


        model.train()
        for i, batch in tqdm(enumerate(train_loader)):
            images, labels = batch
            images = images.to(device).float()
            labels = labels.to(device).float()
            im1 = images[:, 0, :, :, :].float().permute(0, 3, 2, 1)
            im2 = images[:, 1, :, :, :].float().permute(0, 3, 2, 1)

            print("im1", im1.shape)
            raise Exception("breakpoint")

            # print("checking data spec")
            # print(im1.shape, im2.shape, labels.shape)
            # print(images.max(), images.min())

            # plt.imshow(im1[0].cpu().numpy().transpose(0, 1, 2))
            # plt.show()
            # plt.imshow(im2[0].cpu().numpy().transpose(0, 1, 2))
            # plt.show()
            # print(labels)
            # raise Exception("breakpoint")

            optim.zero_grad()
            output1 = model(im1).float()
            output2 = model(im2).float()

            loss = criterion(output1, output2, labels)

            total_loss += loss.item()
            loss_count += 1

            loss.backward()
            optim.step()

            if i % 50 == 0:
                print()
                print("-----------------------------------------------------------")
                print("output1: ", end="")
                print(f"{output1[0].detach().cpu().numpy()}")
                print("output2: ", end="")
                print(f"{output2[0].detach().cpu().numpy()}")
                print(f"output1/2 Cos Sim:")
                print(f"{F.cosine_similarity(output1, output2, dim=1)}")
                # snap cosine sim values to -1 to 1 to check model accuracy
                cosine_sim_binary = F.cosine_similarity(output1, output2, dim=1)
                cosine_sim_binary = torch.where(cosine_sim_binary > 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

                # compare with labels
                print(f"cosine_sim_bin:")
                print(f"{cosine_sim_binary}")
                print(f"correct_counter:")
                correct = np.asarray([1 if cosine_sim_binary[j] == labels[j] else 0 for j in range(len(labels))])
                print(correct)
                print(f"Correct: {correct.sum() / len(labels):.4f}")
                # print(output1[0].detach().cpu().numpy())
                # print(output2[0].detach().cpu().numpy())
                print(f"labels:")
                print(f"{labels}")
                print(f"loss: {loss.item():.4f}")
                print(f"Epoch {epoch + 1}/{epochs}, Iter {i + 1}, Loss: {total_loss / loss_count:.4f}")
                total_loss = 0.0
                loss_count = 0
        # Evaluate on test set
        print(f"Evaluating on test set at epoch {epoch + 1}/{epochs}...")
        for i, data in enumerate(test_loader):
            images, labels = data
            images = images.to(device).float()
            labels = labels.to(device).float()
            im1 = images[:, 0, :, :, :].float().permute(0, 3, 2, 1)
            im2 = images[:, 1, :, :, :].float().permute(0, 3, 2, 1)

            with torch.no_grad():
                output1 = model(im1).float()
                output2 = model(im2).float()

            cosine_sim_binary = F.cosine_similarity(output1, output2, dim=1)
            cosine_sim_binary = torch.where(cosine_sim_binary > 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

            correct = np.asarray([1 if cosine_sim_binary[j] == labels[j] else 0 for j in range(len(labels))])
            test_correct += correct.sum()
            test_count += len(labels)
        print(f"Test Accuracy: {test_correct / test_count:.4f}")
        torch.save(model.state_dict(), f"frame_match_model_epoch_{epoch + 1}.pth")
        print(f"Model saved at epoch {epoch + 1}")

def main4():
    torch.manual_seed(51)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = BasicCNN(320, 240, 3, num_classes=64).to(device)
    model = ConvFormer(img_embedder=embedder, img_embedder_path="frame_match_model_epoch_1.pth").to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    epochs = 10

    sequence_meshability_dataset = FrameSequenceMeshification(Path("/media/SharedStorage/redwood/FramesOnly"))
    test_ratio = 0.1
    test_size = int(len(sequence_meshability_dataset) * test_ratio)
    train_size = len(sequence_meshability_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(sequence_meshability_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    loss_count = 0

    for epoch in range(epochs):
        model.eval()
        correct = 0
        count = 0

        model.train()
        for i, batch in enumerate(train_loader):
            sequences, labels = batch

            labels = labels.to(device).float()
            sequences = sequences.to(device).float()
            # print("sequences shape: ", sequences.shape)
            optim.zero_grad()

            outputs = model(sequences).float()
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            loss_count += 1
            loss.backward()
            optim.step()

            for j in tqdm(range(len(outputs))):
                correct = correct + (outputs[j].argmax() == labels[j].argmax()).item()
                count += len(outputs)

            if i % 50 == 0:
                print()
                print("-----------------------------------------------------------")
                print("total loss:", total_loss / loss_count)
                print("accuracy:", correct / count)
                print("outputs: ", outputs.permute((1, 0)).detach().cpu().numpy())
                print("labels: ", labels.permute((1, 0)).detach().cpu().numpy())
                print(f"Epoch {epoch + 1}/{epochs}, Iter {i + 1}, Loss: {total_loss / loss_count:.4f}")
                total_loss = 0.0
                loss_count = 0
        # Evaluate on test set
        print(f"Evaluating on test set at epoch {epoch + 1}/{epochs}...")
        model.eval()
        test_correct = 0
        test_count = 0
        for i, data in enumerate(test_loader):
            sequences, labels = data
            labels = labels.to(device).float()
            sequences = sequences.to(device).float()

            with torch.no_grad():
                outputs = model(sequences).float()

            for j in range(len(outputs)):
                test_correct += (outputs[j].argmax() == labels[j].argmax()).item()
                test_count += len(outputs)
        print(f"Test Accuracy: {test_correct / test_count:.4f}")
        torch.save(model.state_dict(), f"convformer_model_epoch_{epoch + 1}.pth")

def confformer_test():
    torch.manual_seed(51)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = BasicCNN(320, 240, 3, num_classes=64)
    model = ConvFormer(img_embedder=embedder, img_embedder_path="frame_match_model_epoch_1.pth")
    model.to(device)
    model.load_state_dict(torch.load("convformer_model_epoch_1.pth"))

    sequence_meshability_dataset = FrameSequenceMeshification(Path("/media/SharedStorage/redwood/FramesOnly"))
    test_ratio = 0.1
    test_size = int(len(sequence_meshability_dataset) * test_ratio)
    train_size = len(sequence_meshability_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(sequence_meshability_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    # evaluate on test
    correct = 0
    count = 0
    for i, data in enumerate(test_loader):
        sequences, labels = data
        labels = labels.to(device).float()
        sequences = sequences.to(device).float()

        outputs = model(sequences).float()

        print(f"outputs: {outputs.permute((1, 0)).detach().cpu().numpy()}")
        print(f"labels: {labels.permute((1, 0)).detach().cpu().numpy()}")

        count_corrects = [1 if outputs[j][0] > 0.5 and labels[j][0] > 0.5 or outputs[j][0] < 0.5 and labels[j][0] < 0.5 else 0 for j in range(len(outputs))]
        print(count_corrects)
        print(sum(count_corrects))
        correct += sum(count_corrects)
        count += len(outputs)
    print(f"Test Accuracy: {correct / count:.4f}")


if __name__ == "__main__":
    # main3()
    # main4()
    confformer_test()