"""Checkpoint-compatible XAI/QXAI architectures with a vectorized quantum training path."""
import torch
from torch import nn
from torchvision import models
import pennylane as qml

CLASSES=['AD','CONTROL','PD']
N_QUBITS=4
N_LAYERS=4

class ClassicalCNN(nn.Module):
    def __init__(self,imagenet_state=None):
        super().__init__()
        base=models.efficientnet_b0(weights=None)
        if imagenet_state is not None:base.load_state_dict(imagenet_state)
        for param in base.parameters():param.requires_grad=False
        self.features=base.features
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(1280,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,3))
    def forward(self,images,return_features=False):
        features=self.pool(self.features(images)).flatten(1)
        return features if return_features else self.classifier(features)

quantum_device=qml.device('default.qubit',wires=N_QUBITS)
@qml.qnode(quantum_device,interface='torch',diff_method='backprop')
def quantum_circuit(inputs,weights):
    qml.AngleEmbedding(inputs,wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights,wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(index)) for index in range(N_QUBITS)]

class QuantumHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(1280,N_QUBITS)
        self.q_weights=nn.Parameter(torch.randn(N_LAYERS,N_QUBITS,dtype=torch.float32))
        self.classifier=nn.Sequential(nn.Linear(N_QUBITS,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,features):
        inputs=torch.tanh(self.encoder(features))
        observables=quantum_circuit(inputs,self.q_weights)
        encoded=torch.stack([value.to(dtype=features.dtype) for value in observables],dim=-1)
        return self.classifier(encoded)
