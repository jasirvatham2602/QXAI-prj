# QXAI-prj
Investigates whether a Quantum Explainable AI (QXAI) model that classifies Alzheimer’s disease, Parkinson’s disease, and Control performs better than its XAI classical counterpart. Implements a VQC–CNN (Variational Quantum Circuit–Convolutional Neural Network) and compares XAI and QXAI Saliency Maps (XAI:86.76% and  QXAI:84.56% test accuracy).
# Motivation and Research Question
# Model Architectures 
&emsp;  The XAI model's architecture uses a pretrained EfficientNetB0 backbone with frozen weights to obtain a 1280-length vector; this 1280-length feature vector is then fed into a fully connected hidden layer of 256 neurons with a ReLU activation function. Finally, there is the output layer with 3 neurons with logits to determine if the model predicts Alzheimer's disease, Parkinson's disease, or Control. \
&emsp;  Comparitively the QXAI model's architecture is fairly complex; since quantum circuits can only have a limited number of qubits, a complex architecture is needed to express the features from a MRI scan into a couple of qubits. The QXAI architecture starts with a classical neural network, then goes into a variational quantum circuit (VQC) and final back to a simple neural network to finally obtain 3 output neurons. 
# Explainable AI (Quantum and Classical Saliency Maps)
# Results 
# Statistical Comparison (QXAI vs. XAI with McNemar's Test)
# How to Run
1. Clone the repository with "git clone https://github.com/jasirvatham2602/QXAI-prj" on the terminal, or simply download the contents of the repository as a zip file and extract it on windows
2. Open the QXAI-prj folder on the terminal/command prompt with "cd QXAI-prj"
3. The data has been compressed into a zip folder for convenience. Simply unzip it with "unzip data.zip" on terminal or right click on the data file and unzip it on windows
4. Now that the data file has been unzipped, we must install the needed dependencies with "pip install -r requirements.txt" or "pip3 install -r requirements.txt", if pip doesn't work.
5. Now run the code with "python main.py" or "python3 main.py" (Make sure python is already installed on the computer. If it is not, you can download it at python.org)
# Limitation and Future work
# Acknowledgements 
