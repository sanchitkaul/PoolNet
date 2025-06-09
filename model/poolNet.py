from transformers import ViTModel, ViTImageProcessor
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2 
from dataloader import Scene
from torchvision import models, transforms

class PatchEmbeddingDecoder(nn.Module):
    pass

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=256):
        super().__init__()
        self.patch_size = patch_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, emb_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(inplace=True),
        )

        # Final projection to non-overlapping patch tokens
        self.patch_proj = nn.Conv2d(
            emb_dim,
            emb_dim,
            kernel_size=patch_size // 4,
            stride=patch_size // 4
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.stem(x)        # [B, emb_dim, H/8, W/8]
        x = self.patch_proj(x)  # [B, emb_dim, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, emb_dim]
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)

class SequentialTransformer(nn.Module):
    def __init__(self, image_count, patch_size=8, emb_dim=256, output_type='regression'):
        super().__init__()
        assert output_type in ['embedding', 'regression']
        self.image_count = image_count
        self.emb_dim = emb_dim
        self.output_type = output_type

        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=patch_size, emb_dim=emb_dim)
        self.transformer = TransformerEncoder(emb_dim=emb_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (320 // patch_size) * (240 // patch_size) + 1, emb_dim))

        if output_type == 'embedding':
            self.head = nn.Identity()
        else:  # regression
            self.head = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, 1)
            )

    def forward(self, x):
        B = x.size(0)
        C = 3
        H = 320
        W = 240
        I = self.image_count

        # Reshape to sequence of [B * I, 3, H, W]
        x = x.view(B, I, C, H, W).reshape(B * I, C, H, W)

        x = self.patch_embed(x)  # [B*I, N, emb_dim]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B*I, N+1, emb_dim]
        x = x + self.pos_embed[:, :x.size(1)]

        x = self.transformer(x)  # [B*I, N+1, emb_dim]

        # Take CLS token
        x = x[:, 0]  # [B*I, emb_dim]
        x = x.view(B, I, self.emb_dim)  # [B, I, emb_dim]

        if self.output_type == 'embedding':
            return x  # [B, I, emb_dim]
        else:  # regression
            x = x.mean(dim=1)  # [B, emb_dim]
            return self.head(x)  # [B, 1]

class BasicCNN(nn.Module):
    def __init__(self, input_width=224, input_height=224, input_channels=6, num_classes=1):
        super(BasicCNN, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        # self.bn_cnv1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn_cnv2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn_cnv3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (input_width // 4) * (input_height // 4), 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, num_classes)
        self.bn_fc3 = nn.BatchNorm1d(num_classes)

        self.act = nn.LeakyReLU()
        # self.final = nn.ReLU()  # Set seed for reproducibility
        self.final = nn.Sigmoid()  # Final activation function for output layer
        self.__init_weights()
        # self.__init_weights_2()  # Use He initialization

    def __init_weights_2(self):
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init_weights(self):
        #xavier
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn_cnv1(x)
        # x = self.act(x)

        x = self.conv2(x)
        x = self.bn_cnv2(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn_cnv3(x)
        x = self.act(x)
        x = self.pool(x)

        # x = self.conv4(x)
        # x = self.bn_cnv4(x)
        # x = self.act(x)
        # x = self.pool(x)

        # flatten along non-batched dimension
        x = x.flatten(start_dim=1)  # Flatten all dimensions except the batch dimension
        x = self.fc1(x)
        # x = self.act(x)
        # x = self.bn_fc1(x)

        x = self.fc2(x)
        # x = self.act(x)
        # x = self.bn_fc2(x)

        x = self.fc3(x)
        # x = self.final(x)
        # x = self.bn_fc3(x)

        return x

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
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=2, encoder=None, processor=None, device=torch.device("cpu")):
        super().__init__()

        if encoder is None or processor is None:
            # self.encoder = models.resnet152(pretrained=True)
            # self.encoder.train()
            # self.encoder.fc = nn.Identity()
            # self.processor = transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # # ])
            self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            self.encoder = encoder
            self.processor = processor

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        self.ts = nn.Linear(hidden_dim, 1)
        self.to_output = nn.Linear(hidden_dim, 1)
        self.vr = nn.Linear(hidden_dim, 1)
        self.vt = nn.Linear(hidden_dim, 1)

        self.device = device

    def forward(self, x):
        x1 = x[:, 0:3, :, :]  # First image
        x2 = x[:, 3:6, :, :]  # Second image

        hidden = None
        output = None

        j = 0
        if isinstance(x, Scene):

            while x.has_next():
                # print(f"Processing scene: {x.scene_id}, image {j + 1}, cuda usage: {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.2f} GB")
                data = x.get_next()
                image, label, self.scene_id, key = data
                # img_input = self.processor(images=image, return_tensors="pt").to(self.device)
                image = Image.fromarray(image)
                img_input = self.processor(image).unsqueeze(0).to(self.device)  # Add batch dimension
                img_encodings = self.encoder(img_input).unsqueeze(0)  # Add batch dimension
                # img_encodings = self.encoder(**img_input).last_hidden_state[:, 0, :]  # CLS token
                sequence = torch.stack([img_encodings, img_encodings], dim=1)
                output, hidden = self.lstm(img_encodings, hidden)
                j += 1
        else:
            pil_x1 = [transforms.ToPILImage()(img.cpu()) for img in x1]
            pil_2x = [transforms.ToPILImage()(img.cpu()) for img in x2]
            processed_x1 = self.processor(pil_x1, return_tensors="pt")["pixel_values"].to("cuda")  # Process the input tensor
            processed_x2 = self.processor(pil_2x, return_tensors="pt")["pixel_values"].to("cuda")  # Process the input tensor


            
            embedding1 = self.encoder(processed_x1).last_hidden_state[:, 0, :]
            embedding2 = self.encoder(processed_x2).last_hidden_state[:, 0, :]

            # processed_x1 = torch.stack([self.processor(transforms.ToPILImage()(img)) for img in x1]) # Process the input tensor
            # processed_x2 = torch.stack([self.processor(transforms.ToPILImage()(img)) for img in x2]) # Process the input tensor
            # embedding1 = self.encoder(x1)  # Add batch dimension
            # embedding2 = self.encoder(x2)  # Add batch dimension
            sequence = torch.stack([embedding1, embedding2], dim=1)
            output, hidden = self.lstm(sequence,)
        h_n = hidden[0][-1]
        c_n = hidden[1][-1]

        # NOTE: Might need to use output in finl prediction? or both? tbd

        # elif x.dim() == 2:
        #     x = x.unsqueeze(0)  # Add batch dimension
        # _, (hn, _) = self.lstm(x)
        # last_hidden = hn[-1]  # Last layer's hidden state
        # print(f"Last hidden state shape: {h_n.shape}, c_n shape: {c_n.shape}")
        return self.vr(h_n)
        return {
            "ts": torch.sigmoid(self.ts(h_n)),
            "to_output": self.to_output(h_n),
            "vr": self.vr(h_n),
            "vt": self.vt(h_n)
        }
    
    def inference(self, x):
        # method to inference sequentially
        pass
    
class LSTMHead2(nn.Module):
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

