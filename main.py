import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
from torchvision import datasets, transforms
from collections import defaultdict 

NUM_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 10
IMAGE_SIZE = 224

N_QUBITS = 4 
N_LAYERS = 4 # changed from 3 to 4
## Quantum model's Epochs is less than the XAI model because the XAI model is already trained
Q_EPOCHS = 5 

DATA_DIR = "./data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['AD', 'CONTROL', 'PD']

base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

full_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}",
    transform=base_transform
)

print("Classes:", full_dataset.classes)

def stratified_split(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    np.random.seed(seed)

    targets = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    train_idx, val_idx, test_idx = [], [], []

    for class_id in np.unique(targets):
        class_indices = indices[targets == class_id]
        np.random.shuffle(class_indices)

        n_total = len(class_indices)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_idx.extend(class_indices[:n_train])
        val_idx.extend(class_indices[n_train:n_train + n_val])
        test_idx.extend(class_indices[n_train + n_val:])

    return train_idx, val_idx, test_idx
train_idx, val_idx, test_idx = stratified_split(full_dataset)

train_ds = Subset(full_dataset, train_idx)
val_ds   = Subset(full_dataset, val_idx)
test_ds  = Subset(full_dataset, test_idx)

train_ds.dataset.transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

val_ds.dataset.transform = base_transform
test_ds.dataset.transform = base_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
from collections import Counter

def count_subset(subset):
    return Counter([subset.dataset.targets[i] for i in subset.indices])

print("Train distribution:", count_subset(train_ds))
print("Val distribution:", count_subset(val_ds))
print("Test distribution:", count_subset(test_ds))

print("Class names:", full_dataset.classes)
class ClassicalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.efficientnet_b0(pretrained=True)
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
def train_classical(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    scheduler
):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # -------- TRAIN --------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # -------- VALIDATE --------
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        # -------- LOG --------
        print(
            f"[Classical] Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} |"
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(val_loss)
    return train_losses, train_accs, val_losses, val_accs
cnn = ClassicalCNN().to(DEVICE)
optimizer = torch.optim.Adam(cnn.classifier.parameters(), lr=1e-3, weight_decay = 1e-4) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience = 3)
loss_fn = nn.CrossEntropyLoss()
train_losses, train_accs, val_losses, val_accs = train_classical(model=cnn,train_loader=train_loader,val_loader=val_loader,optimizer=optimizer,loss_fn=loss_fn,device=DEVICE,epochs=EPOCHS, scheduler=scheduler)
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
                torch.as_tensor(v, dtype=torch.float32, device=DEVICE)
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
def early_stopping(train_loss, validation_loss, min_delta, tolerance):
    counter = 0
    if (validation_loss - train_loss) > min_delta:
        counter +=1
        if counter >= tolerance:
          return True
def train_qxai(
    cnn,
    qxai_model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs
):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # ================= TRAIN =================
        qxai_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Freeze CNN
            with torch.no_grad():
                features = cnn(images, return_features=True)

            optimizer.zero_grad()
            outputs = qxai_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ================= VALIDATE =================
        qxai_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                features = cnn(images, return_features=True)
                outputs = qxai_model(features)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # ================= LOG =================
        print(
            f"[QXAI] Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} |"
            f"Val-Train Loss: {val_loss-train_loss:.4f} "
        )
        if early_stopping(train_loss, val_loss, min_delta=0.2, tolerance = 3): # val_los - train_loss must be < 0.2 
              print("We are at epoch:", i)
              break

    return train_losses, train_accs, val_losses, val_accs
qxai_model = QuantumHybrid().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(qxai_model.parameters(), lr=1e-3)
qxai_train_losses, qxai_train_accs, qxai_val_losses, qxai_val_accs = train_qxai(cnn=cnn,qxai_model=qxai_model,train_loader=train_loader,val_loader=val_loader,optimizer=optimizer,criterion=criterion,device=DEVICE,epochs=Q_EPOCHS)
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train")
plt.plot(epochs_range, val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("XAI Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label="Train")
plt.plot(epochs_range, val_accs, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("XAI Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
epochs_range = range(1, Q_EPOCHS + 1)

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, qxai_train_losses, label="Train")
plt.plot(epochs_range, qxai_val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("QXAI Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, qxai_train_accs, label="Train")
plt.plot(epochs_range, qxai_val_accs, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("QXAI Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
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
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    saliency = saliency.squeeze().cpu().numpy()

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.imshow(saliency, cmap="hot", alpha=0.5)
    plt.axis("off")
    plt.title(title)
    plt.show()
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
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    saliency = saliency.squeeze().cpu().numpy()

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.imshow(saliency, cmap="hot", alpha=0.5)
    plt.axis("off")
    plt.title(title)
    plt.show()
def show_xai_comparison(image, c_sal, q_sal,
                        c_title="Classical XAI",
                        q_title="Quantum XAI"):
    """
    image: (1, C, H, W)
    c_sal, q_sal: (1, H, W)
    """

    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
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
    plt.show()
cnn.eval()
qxai_model.eval()

class_names = ["AD", "CONTROL", "PD"]

NUM_CLASSES = 3


for batch_idx, (x, y) in enumerate(train_loader):
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    # Take ONE sample from the batch
    image = x[:1]
    true_label = y[0].item()
    class_name = class_names[true_label]
 
        

    # -------- Classical --------
    c_logits = cnn(image)
    c_label = c_logits.argmax(dim=1)[0].item()
    c_sal = classical_saliency(cnn, image, c_label)

    # -------- Quantum --------
    q_features = cnn(image, return_features=True)
    q_logits = qxai_model(q_features)
    q_label = q_logits.argmax(dim=1)[0].item()
    q_sal = quantum_image_saliency(cnn, qxai_model, image, q_label)

    # Seeing the predictions from QXAI and XAI models & their Saliency Maps
    if c_label == q_label and q_label == true_label:
        print('Both models are correct')
    elif c_label == true_label and q_label != true_label: # XAI model is correct
        print('XAI model is correct')
    elif c_label != true_label and q_label == true_label: # QXAI model is correct
        print('QXAI model is correct')
    else:
        print('neither model is correct') 

    # Displaying Saliency Maps side by side. 
    show_xai_comparison(
        image,
        c_sal,
        q_sal,
        c_title=f"Classical XAI | Pred: {class_names[c_label]} | True: {class_name}",
        q_title=f"Quantum XAI | Pred: {class_names[q_label]} | True: {class_name}"
    )

# Displaying Confusion Matrix for classical XAI model 
def display_confusion_matrix(cm, title_of_cm='Confusion Matrix for classical XAI model', label_x = 'Actual Label', label_y = 'Predicted Label', xtick = class_names, ytick = class_names):
    plt.figure(figsize=(7, 5)) # Optional: Adjust figure size
    # sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=.5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, yticklabels=ytick, xticklabels=xtick)
    plt.ylabel(label_y, fontsize=14)
    plt.xlabel(label_x, fontsize=14)
    plt.title(title_of_cm, fontsize=18)
    plt.show()
# Confusion matrix for the Classical XAI model
def get_confusion_matrix_classical(model, dataloader):
    model.eval()
    confusion_matrix_classical_xai = [[0]* NUM_CLASSES for i in range(NUM_CLASSES)] 
    cnt = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            # X is 8 image as a tensor (batch_size)
            # y is 8 true_labels [0, 0 ... 0 ] (batch_size)
            # preds is 8 prd_labels from model [0, 0 ... 0 ] (batch_size)
            # outputs is 8 rows , 3 columns (with a number non-softmax e.g. [5, -1 , -3]) not probability, max num's index indicates the pred 
            preds = outputs.argmax(dim=1)             
            for i in range(len(list(preds))): 
                # preds & y may not always have len equal to batch_size
                # bc the amount of test data is not always divisible by batch_size  
                pred_list = list(preds)
                y_list = list(y)
                confusion_matrix_classical_xai[pred_list[i]][y_list[i]] += 1
                if pred_list[i] == y_list[i]:
                    cnt += 1
                total += 1 
    accuracy = cnt / total 
    return confusion_matrix_classical_xai, accuracy 
def get_confusion_matrix_quantum(model, dataloader):
    model.eval()
    confusion_matrix_classical_xai = [[0]* NUM_CLASSES for i in range(NUM_CLASSES)] 
    cnt = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # outputs = model(x)
            # X is 8 image as a tensor (batch_size)
            # y is 8 true_labels [0, 0 ... 0 ] (batch_size)
            # preds is 8 prd_labels from model [0, 0 ... 0 ] (batch_size)
            # outputs is 8 rows , 3 columns (with a number non-softmax e.g. [5, -1 , -3]) not probability, max num's index indicates the pred 
            
            # preds = outputs.argmax(dim=1)             
            features = cnn(x, return_features=True)
            preds = qxai_model(features).argmax(dim=1)
            for i in range(len(list(preds))): 
                # preds & y may not always have len equal to batch_size
                # bc the amount of test data is not always divisible by batch_size  
                pred_list = list(preds)
                y_list = list(y)
                confusion_matrix_classical_xai[pred_list[i]][y_list[i]] += 1
                if pred_list[i] == y_list[i]:
                    cnt += 1  
                total += 1
    accuracy = cnt / total 
    return confusion_matrix_classical_xai, accuracy 


cm_xai, acc_xai = get_confusion_matrix_classical(cnn, test_loader)
cm_qxai, acc_qxai = get_confusion_matrix_quantum(qxai_model, test_loader)
print(acc_xai)
print(acc_qxai)
print(f"XAI Test acc: {100*acc_xai:.2f}%")
print(f"QXAI Test acc: {100*acc_qxai:.2f}%")

display_confusion_matrix(cm_xai)
display_confusion_matrix(cm_qxai, 'Confusion Matrix for QXAI model')
cm_train_xai, acc_train_xai = get_confusion_matrix_classical(cnn, train_loader)
cm_train_qxai, acc_train_qxai = get_confusion_matrix_quantum(qxai_model, train_loader) 
print(f"XAI Train acc: {100*acc_train_xai:.2f}%")
print(f"QXAI Train acc: {100*acc_train_qxai:.2f}%")
display_confusion_matrix(cm_train_xai, 'Confusion Matrix for XAI model of training dataset')
display_confusion_matrix(cm_train_qxai, 'Confusion Matrix for QXAI model of training dataset')
print(acc_train_xai)
print(acc_train_qxai)
# Comparing the QXAI model with the XAI model


'''

        [    XAI correct, XAI incorrect 
 Q correct           [a, b]
 Q incorrect         [c, d]
        ]


'''

def get_confusion_matrix_classical_vs_quantum(cnnmodel, qmodel, dataloader):
    cnnmodel.eval()
    qmodel.eval()
    confusion_matrix_classical_vs_quantum = [[0]* 2 for i in range(2)] 
    cnt = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
                # X is 8 image as a tensor (batch_size)
                # y is 8 true_labels [0, 0 ... 0 ] (batch_size)
                # preds is 8 prd_labels from model [0, 0 ... 0 ] (batch_size)
    #             # outputs is 8 rows , 3 columns (with a number non-softmax e.g. [5, -1 , -3]) not probability, max num's index indicates the pred 
    #         # Classical model's predictions
            outputs_classical = cnnmodel(x)
            preds_classical = outputs_classical.argmax(dim=1)

            # Quantum model's prediction 
            features = cnnmodel(x, return_features=True)
            preds_quantum = qxai_model(features).argmax(dim=1)
            
            for i in range(len(list(preds_classical))): 
                assert len(list(preds_classical)) == len(list(preds_quantum)) 
                # preds & y may not always have len equal to batch_size
                # bc the amount of test data is not always divisible by batch_size  
                preds_classical_list = list(preds_classical)
                preds_quantum_list = list(preds_quantum)
                y_list = list(y)
                if preds_classical_list[i] == preds_quantum_list[i] and y_list[i] == preds_quantum_list[i]:
                    # both models got it correct 
                    confusion_matrix_classical_vs_quantum[0][0] += 1
                elif preds_classical_list[i] != preds_quantum_list[i] and y_list[i] == preds_quantum_list[i]:
                    # Quantum model correct & Classical model incorrect 
                    confusion_matrix_classical_vs_quantum[0][1] += 1
                elif preds_classical_list[i] != preds_quantum_list[i] and y_list[i] == preds_classical_list[i]:
                    # Quantum model incorrect & Classical model correct 
                    confusion_matrix_classical_vs_quantum[1][0] += 1
                else:
                    # Quantum model incorrect & Classical model incorrect 
                    confusion_matrix_classical_vs_quantum[1][1] += 1
    return confusion_matrix_classical_vs_quantum 
cm_classical_vs_quantum = get_confusion_matrix_classical_vs_quantum(cnn, qxai_model, test_loader)

display_confusion_matrix(cm_classical_vs_quantum, 'Confusion Matrix of QXAI model vs. XAI model', label_y = 'QXAI', label_x = 'XAI', xtick=['Correct', 'Incorrect'],  ytick=['Correct', 'Incorrect'])