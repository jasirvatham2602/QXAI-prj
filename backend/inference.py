from model_bundle import load_bundle
BUNDLE = load_bundle()
import time
from fastapi import FastAPI, UploadFile, File
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch
from PIL import Image
import pennylane as qml
import matplotlib.pyplot as plt
from fastapi.staticfiles import StaticFiles
# from sqlalchemy import Column, Integer, String
# from database import Base
import uvicorn
NUM_CLASSES = 3
IMAGE_SIZE = 224

# Quantum
N_QUBITS = 4
N_LAYERS = 4
# dir = 'uploaded_images'
# try:
#     os.mkdir(dir)
# except FileExistsError:
#     print(f'dir {dir} already exists')
# dir2 = 'saliency_maps'
# try:
#     os.mkdir(dir2)
# except FileExistsError:
#     print(f'dir {dir2} already exists')
UPLOAD_DIR = '/tmp/uploaded_images'
SALIENCY_DIR = '/tmp/saliency_maps'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SALIENCY_DIR, exist_ok=True)

# Databases
# class User(Base):
#     __tablename__ = 'users'
#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True)
#     hashed_password = Column(String)

# Loading the trained pth model
class ClassicalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.efficientnet_b0(weights=None)
        for p in base.parameters():
            p.requires_grad = False # freezes pretrained model

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if return_features:
            return x

        return self.classifier(x)
class QuantumHybrid(nn.Module):
    def __init__(self):
        super().__init__()

        # Classical → Quantum encoder
        self.encoder = nn.Linear(1280, N_QUBITS)

        # Quantum parameters
        self.q_weights = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS, dtype=torch.float32)
        )

        # Classical head
        self.classifier = nn.Sequential(
            nn.Linear(N_QUBITS, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, features):
        """
        features: (batch, 1280)
        """

        # Encode to quantum dimension
        x = self.encoder(features)
        x = torch.tanh(x)  # bound rotation angles

        batch_size = x.size(0)
        q_outputs = []

        for i in range(batch_size):
            sample = x[i].to(dtype=torch.float32)
            q_out = quantum_circuit(sample, self.q_weights)

            # Ensure torch tensor
            q_out = torch.stack([
                torch.as_tensor(v, dtype=torch.float32, device=device)
                for v in q_out
            ])

            q_outputs.append(q_out)

        q_outputs = torch.stack(q_outputs)  # (batch, N_QUBITS)
        return self.classifier(q_outputs)
dev = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    inputs: (N_QUBITS,)
    weights: (N_LAYERS, N_QUBITS)
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
] + ([transforms.Normalize(BUNDLE['mean'], BUNDLE['std'])] if BUNDLE['mean'] else []))

# device for model/tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xai_model = ClassicalCNN()
xai_model.load_state_dict(torch.load(BUNDLE['paths']['classical'], map_location=device))
xai_model.to(device)
xai_model.eval()

qxai_model = QuantumHybrid().to(device)
qxai_model.load_state_dict(torch.load(BUNDLE['paths']['quantum'], map_location=device))
qxai_model.eval()

class_names = ['AD', 'CONTROL', 'PD']
# Saliency Maps

def classical_saliency(model, image, label):
    model.zero_grad()
    image = image.clone().detach().requires_grad_(True)

    output = model(image)
    score = output[0, label]
    score.backward()

    saliency = image.grad.abs()
    saliency, _ = torch.max(saliency, dim=1)  # collapse RGB → 1 channel
    return saliency
def show_classical_saliency(image, saliency, title="Classical Saliency"):
    global img_cm_cnt
    img_cm_cnt += 1
    image = image.detach().squeeze().cpu()
    if BUNDLE['mean']:
        image = image * torch.tensor(BUNDLE['std'])[:, None, None] + torch.tensor(BUNDLE['mean'])[:, None, None]
    image = image.permute(1, 2, 0).clamp(0, 1).numpy()
    saliency = saliency.squeeze().cpu().numpy()

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.imshow(saliency, cmap="hot", alpha=0.5)
    plt.axis("off")
    plt.title(title)
    plt.savefig(f"classical_saliency_map_{img_cm_cnt}", dpi = 300, bbox_inches="tight")
    # plt.show()
def quantum_saliency(q_model, features, label):
    q_model.zero_grad()
    features = features.clone().detach().requires_grad_(True)

    output = q_model(features)
    score = output[0, label]
    score.backward()

    saliency = features.grad.abs()  # (1, 1280)
    return saliency
def quantum_image_saliency(cnn, q_model, image, label):
    cnn.zero_grad()
    q_model.zero_grad()

    image = image.clone().detach().requires_grad_(True)

    # Forward pass
    features = cnn(image, return_features=True)
    features.retain_grad()

    output = q_model(features)
    score = output[0, label]
    score.backward()

    # Chain-rule saliency: dQ/dImage
    saliency = image.grad.abs()
    saliency, _ = torch.max(saliency, dim=1)  # collapse channels

    return saliency
def show_quantum_saliency(image, saliency, title="Quantum Saliency"):
    global img_cm_cnt
    img_cm_cnt += 1
    image = image.detach().squeeze().cpu()
    if BUNDLE['mean']:
        image = image * torch.tensor(BUNDLE['std'])[:, None, None] + torch.tensor(BUNDLE['mean'])[:, None, None]
    image = image.permute(1, 2, 0).clamp(0, 1).numpy()
    saliency = saliency.squeeze().cpu().numpy()

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.imshow(saliency, cmap="hot", alpha=0.5)
    plt.axis("off")
    plt.title(title)
    plt.savefig(f"quantum_saliency_map_{img_cm_cnt}", dpi = 300, bbox_inches="tight")
    # plt.show()
# img_cnt = 0
img_cm_cnt = 0
def save_xai_qxai_saliency_map_comparison(image, c_sal, q_sal,
                        c_title="Classical XAI",
                        q_title="Quantum XAI", dir=None, file_name=None):
    """
    image: (1, C, H, W)
    c_sal, q_sal: (1, H, W)
    """
    if dir is None:
        dir = SALIENCY_DIR
    global img_cm_cnt
    img_cm_cnt += 1
    print(f"global_img_cm_cnt {img_cm_cnt}")
    image = image.detach().squeeze().cpu()
    if BUNDLE['mean']:
        image = image * torch.tensor(BUNDLE['std'])[:, None, None] + torch.tensor(BUNDLE['mean'])[:, None, None]
    image = image.permute(1, 2, 0).clamp(0, 1).numpy()
    c_sal = c_sal.squeeze().cpu().numpy()
    q_sal = q_sal.squeeze().cpu().numpy()

    # Normalize saliency maps
    c_sal = (c_sal - c_sal.min()) / (c_sal.max() - c_sal.min() + 1e-8)
    q_sal = (q_sal - q_sal.min()) / (q_sal.max() - q_sal.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Classical
    axes[0].imshow(image, cmap="gray")
    axes[0].imshow(c_sal, cmap="hot", alpha=0.5)
    axes[0].set_title(c_title)
    axes[0].axis("off")

    # Quantum
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(q_sal, cmap="hot", alpha=0.5)
    axes[1].set_title(q_title)
    axes[1].axis("off")

    plt.tight_layout()
    index = file_name.index('.')
    file_name = file_name[0: index]
    output_file_name = f"SaliencyMaps_{img_cm_cnt}_{file_name}.png"
    os_dir = os.path.join(dir, output_file_name)
    # os_dir2= os.path.join(dir2, f"SaliencyMaps#{img_cm_cnt}_{file_name}.png")
    plt.savefig(os_dir, dpi=300, bbox_inches='tight')
    # plt.savefig(os_dir2, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    return os_dir, output_file_name




def analyze(file_location, timings=None):
    timings = timings if timings is not None else {}
    started = time.perf_counter()
    img = Image.open(file_location).convert('RGB')
    with torch.no_grad():
        # transform and add batch dimension
        transformed_img = base_transform(img).unsqueeze(0).to(device)
        ''' main3.py file

            # -------- Classical --------
            c_logits = cnn(image)
            c_label = c_logits.argmax(dim=1)[0].item()
            c_sal = classical_saliency(cnn, image, c_label)

            # -------- Quantum --------
            q_features = cnn(image, return_features=True)
            q_logits = qxai_model(q_features)
            q_label = q_logits.argmax(dim=1)[0].item()
            q_sal = quantum_image_saliency(cnn, qxai_model, image, q_label)
        '''
        timings['preprocessing_seconds'] = time.perf_counter() - started
        classical_started = time.perf_counter()
        # output from XAI model
        output = xai_model(transformed_img)
        probabilities = nn.functional.softmax(output, dim=1)
        predicted_class = int(torch.argmax(probabilities, dim=1).item())
        output_list = output.squeeze(0).cpu().tolist()
        prob_list = probabilities.squeeze(0).cpu().tolist()

        # classical_saliency(xai_model, transformed_img, c_label)


        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings['classical_seconds'] = time.perf_counter() - classical_started
        quantum_started = time.perf_counter()
        # output from QXAI model
        features = xai_model(transformed_img, return_features=True)
        q_output = qxai_model(features)
        q_probabilities = nn.functional.softmax(q_output, dim=1)
        q_predicted_class = int(torch.argmax(q_probabilities, dim=1).item())
        q_output_list = q_output.squeeze(0).cpu().tolist()
        q_prob_list = q_probabilities.squeeze(0).cpu().tolist()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['quantum_seconds'] = time.perf_counter() - quantum_started
    saliency_started = time.perf_counter()
    prob_list_rounded = [round(prob, 3) for prob in prob_list]
    q_prob_list_rounded = [round(prob, 3) for prob in q_prob_list]
    with torch.enable_grad():
        c_label = output.argmax(dim=1)[0].item()
        c_sal = classical_saliency(xai_model, transformed_img, c_label)
        q_label = q_output.argmax(dim=1)[0].item()
        q_sal = quantum_image_saliency(xai_model, qxai_model, transformed_img, q_label)
        saliency_maps_path, output_file_name = save_xai_qxai_saliency_map_comparison(
            transformed_img,
            c_sal,
            q_sal,
            c_title=f"Classical XAI | Pred: {class_names[c_label]} \n Prob: {prob_list_rounded} \n file name: analysis.png",
            q_title=f"Quantum XAI | Pred: {class_names[q_label]} \n Prob: {q_prob_list_rounded} \n file name: analysis.png",
            file_name = "analysis.png"
        )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timings['saliency_seconds'] = time.perf_counter() - saliency_started
    return {"classical": {"label": class_names[predicted_class], "probabilities": prob_list}, "quantum": {"label": class_names[q_predicted_class], "probabilities": q_prob_list}}, saliency_maps_path
