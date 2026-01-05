# QXAI-prj
Investigates whether a Quantum Explainable AI (QXAI) model that classifies Alzheimer’s disease, Parkinson’s disease, and Control performs better than its XAI classical counterpart. Implements a VQC–CNN (Variational Quantum Circuit–Convolutional Neural Network) and compares XAI and QXAI Saliency Maps (XAI:86.76% and  QXAI:84.56% test accuracy).
# Motivation and Research Question
# Model Architectures 
&emsp;  The XAI model's architecture uses a pretrained EfficientNetB0 backbone with frozen weights to obtain a 1280-length vector; this 1280-length feature vector is then fed into a fully connected hidden layer of 256 neurons with a ReLU activation function. Finally, there is the output layer with 3 neurons with logits to determine if the model predicts Alzheimer's disease, Parkinson's disease, or Control
![VQC Diagram](https://github.com/jasirvatham2602/QXAI-prj/VQC.png)
&emsp;  The QXAI model employs a quantum-classical hybrid architecture. It starts with a classical neural network, then goes into a variational quantum circuit (VQC), and finally back to a simple neural network to obtain 3 output neurons. The QXAI architecture begins with the EfficientNetB0 backbone to obtain a 1280-length feature vector. Simulating 1280 qubits is infeasible due to limitations of quantum technology. Therefore, this feature is passed into a fully connected hidden layer with 4 neurons with a tanh activation function. Now we have 4 numbers between -1 and 1 to represent the features of the MRI scan. Quantum encoding then converts these numbers to a qubit state on the Bloch sphere; specifically, each qubit $| 0 \rangle$ state is rotated with the Ry gate by the $x_i$. A quantum layer consists of rotations of each of the 4 qubits by a certain amount  $\theta_1$, $\theta_2$, $\theta_3$, and $\theta_4$, which are trainable weights. Then a CNOT gate is applied to neighboring qubits to create entanglement, where the states of the qubits now depend on each other. This quantum layer is repeated 3 more times with 12 more trainable weights. The last part of the Quantum circuit involves measuring the expectation of the Pauli-Z observable, $\langle Z \rangle$, applied to each qubit. This is then fed into a linear fully connected hidden layer with 64 neurons  with a ReLU activation function, and finally, 3 output neurons.

$$ | \psi \rangle = \alpha |0\rangle +\beta |1\rangle = \begin{bmatrix} \alpha \\
\beta \end{bmatrix}  $$ 
$$  P(0) = |\alpha|^2 $$ 
$$ P(1) = |\beta|^2 $$
$$ R_y(\theta) = \begin{bmatrix} cos(\frac{\theta}{2}) & -sin(\frac{\theta}{2}) \\
sin(\frac{\theta}{2}) & cos(\frac{\theta}{2})
\end{bmatrix}$$ 
$$ R_y(\theta)|0\rangle = \begin{bmatrix} cos(\frac{\theta}{2}) & -sin(\frac{\theta}{2}) \\
sin(\frac{\theta}{2}) & cos(\frac{\theta}{2})
\end{bmatrix} 
\begin{bmatrix} 1 \\
0 \end{bmatrix} = \begin{bmatrix} cos(\frac{\theta}{2}) \\
sin(\frac{\theta}{2})
\\end{bmatrix}
$$
$$ Z = \begin{bmatrix} 1 & 0 \\
0 & -1 \end{bmatrix} $$

$$ \langle \psi | = \begin{bmatrix} \bar{\alpha} & \bar {\beta} \\ \end{bmatrix} $$
$$ \langle \psi | Z = \begin{bmatrix} \bar{\alpha} & \bar{\beta} \\ \end{bmatrix} \begin{bmatrix} 1 & 0 \\
0 & -1 \end{bmatrix}  =  \begin{bmatrix} \bar{\alpha} \\
-\bar{\beta}
\end{bmatrix}
$$
$$ \langle Z \rangle = \langle \psi | Z | \psi \rangle =  \begin{bmatrix} \bar{\alpha} 
-\bar{\beta}
\end{bmatrix}
\begin{bmatrix} \alpha \\
\beta 
\end{bmatrix} = \alpha * \bar{\alpha} - \beta * \bar{\beta} = |\alpha|^2 - |\beta|^2 = P(0) - P(1) 
$$
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
