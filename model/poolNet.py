from transformers import ViTModel, ViTImageProcessor
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2 

# Linear  head
class PoolNetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ts = nn.Linear(768, 1)
        self.to_output = nn.Linear(768, 1)
        self.vr = nn.Linear(768, 1)
        self.vt = nn.Linear(768, 1)

    def forward(self, x):
        return {
            "ts": torch.sigmoid(self.ts(x)),
            "to_output": self.to_output(x),
            "vr": self.vr(x),
            "vt": self.vt(x)
        }

#LSTM head
class LSTMHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        self.ts = nn.Linear(hidden_dim, 1)
        self.to_output = nn.Linear(hidden_dim, 1)
        self.vr = nn.Linear(hidden_dim, 1)
        self.vt = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        _, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]  # Last layer's hidden state
        return {
            "ts": torch.sigmoid(self.ts(last_hidden)),
            "to_output": self.to_output(last_hidden),
            "vr": self.vr(last_hidden),
            "vt": self.vt(last_hidden)
        }

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)  
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < num_frames:
        raise ValueError(f"Video has only {total} frames, can't extract {num_frames}.")

    frame_idxs = np.linspace(0, total - 1, num_frames).astype(int) 
    frames = []

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        pil_img = Image.fromarray(rgb_frame).resize((224, 224))
        frames.append(pil_img)

    cap.release()
    return frames

if __name__ == "__main__":
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    head = LSTMHead()
    print(f"Model head: {type(head)}")

    video_path = "./test-videos/00001.mp4" 
    frames = extract_frames(video_path, num_frames=8)

    #dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    #inputs = processor(images=dummy_image, return_tensors="pt")

    cls_vectors = []

    with torch.no_grad():
        for frame in frames:
            inputs = processor(images=frame, return_tensors="pt")
            outputs = vit(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]  
            cls_vectors.append(cls_token.squeeze(0))

    sequence = torch.stack(cls_vectors)
    avg_vector = torch.mean(sequence, dim=0)  # shape: [768]

    # For result using linear model
    # with torch.no_grad():
        #preds = head(avg_vector)

    with torch.no_grad():
        preds = head(sequence)

    ts_val = preds['ts'].squeeze().item()
    to_val = preds['to_output'].squeeze().item()
    vr_val = preds['vr'].squeeze().item()
    vt_val = preds['vt'].squeeze().item()

    print("\n Predictions:")
    print(f"Ts (COLMAP success): {ts_val:.3f}")
    print(f"To (% frames used): {to_val:.3f}")
    print(f"Vr (rotation diversity): {vr_val:.3f}")
    print(f"Vt (translation diversity): {vt_val:.3f}")

