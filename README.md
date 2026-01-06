# QXAI-prj
Investigates whether a Quantum Explainable AI (QXAI) model that classifies Alzheimer’s disease (AD), Parkinson’s disease (PD), and Control performs better than its XAI classical counterpart. Implements a VQC–CNN (Variational Quantum Circuit–Convolutional Neural Network) and compares XAI and QXAI Saliency Maps (XAI:89.46% and  QXAI:86.52% test accuracy).
# Motivation 
In december 4th, 2025, my research on the use of Transfer Learning, Ensemble Learning, and Explainable AI to produce precise and interpretable diagnoses, building trust between neurologists and AI, was published in the American Journal of Student Research. The proposed Ensemble Learning model received an accuracy of 97.04% using classical neural networks. After learning Quantum Computing from CSI 4900 at Oakland University, I wanted to learn about how Quantum Computing is integrated with classical neural networks to produce a Quantum-classical hybrid. Therefore, I explored how to design a Quantum AI architecture to classify AD, PD, and Control. Furthermore, I produced Quantum Saliency Maps to allow side-by-side comparisons between the XAI and QXAI models. 
# Model Architectures 
&emsp;  The XAI model's architecture uses a pretrained EfficientNetB0 backbone with frozen weights to obtain a 1280-length vector; this 1280-length feature vector is then fed into a fully connected hidden layer of 256 neurons with a ReLU activation function. Finally, there is the output layer with 3 neurons with logits to determine if the model predicts Alzheimer's disease, Parkinson's disease, or Control. 

 
  <img src="https://github.com/jasirvatham2602/QXAI-prj/blob/main/VQC.png" width="500" />  
&emsp; &emsp;  &emsp; The VQC diagram was made with Qiskit




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
\end{bmatrix} = \alpha * \bar{\alpha} - \beta * \bar{\beta} = |\alpha|^2 - |\beta|^2 = P(0) - P(1) $$
# Explainable AI (Quantum and Classical Saliency Maps)

<img src="https://github.com/jasirvatham2602/QXAI-prj/blob/main/sidebysideSaliencyMap.png" width="100%" /> 

Above is a Classical and Quantum Saliency Map produced by the models side by side. Both of the models produced the correct diagnosis of PD, but highlighted slightly different regions. Given that both architectures use the same 1280-length feature vector, the highlighted features will be similar, which is the case. However, the Quantum Saliency Map appears to be dimmer, focusing mainly on a certain region on the bottom half of the MRI scan, while the Classical Saliency Map produced was brighter and highlighted multiple regions. 

<img src="https://github.com/jasirvatham2602/QXAI-prj/blob/main/sidebysideSaliencyMap2.png" width="100%" /> 

Above is another side-by-side Saliency Maps of the QXAI and XAI models. However, the models disagreed on their diagnoses; the XAI model correctly predicted CONTROL, while the QXAI model predicted AD. The XAI model highlighted multiple regions confidently; specifically, it highlighted the middle region of the MRI scan in bright yellow. However, the QXAI model didn't really highlight much, possibly indicating that it may not have been confident in its diagnosis. 

# Results 
<img src="https://github.com/jasirvatham2602/QXAI-prj/blob/main/XAI_training.png" width="100%" /> 
<img src="https://github.com/jasirvatham2602/QXAI-prj/blob/main/QXAI_training.png" width="100%" /> 

The QXAI model was trained for 5 epochs, while the XAI model was trained for 10 epochs; when the models were trained for higher epochs, overfitting occurred. The QXAI model needed less time to train since it has fewer trainable parameters in its architecture, with only 16 trainable parameters in its 4 quantum layers and a final hidden layer of 64 neurons, fewer than its XAI counterpart. Both QXAI and XAI's train loss and validation curves generally decreased during training. A increasing validation curve and a decreasing training loss curve indicate that the model is likely overfitting, which is when a model studies the training data too well, performing worse on unseen data. Therefore, the models are not overfit. Both QXAI and XAI training and validation accuracy graphs show an upward trend.  
<div>
  <img src="https://github.com/jasirvatham2602/QXAI-prj/blob/main/cm_XAI.png" width="500" /> 
  <img src="https://github.com/jasirvatham2602/QXAI-prj/blob/main/cm_QXAI.png" width="500" /> 
</div>
Above is the Confusion Matrices of QXAI and XAI models for the test dataset. The QXAI model received an accuracy of 86.52%, while the XAI model received an accuracy of 89.46%. Both models performed well at identifying PD. Both models struggle with the classification of AD and CONTROL. The QXAI correctly identified 191 AD, while XAI correctly identified 179 AD; the XAI model correctly identified 134 CONTROL cases, while the QXAI model correctly classified 110 CONTROL cases. Overall, both models performed well, with the XAI model performing better.   

# Statistical Comparison (QXAI vs. XAI with McNemar's Test)
<img src="https://github.com/jasirvatham2602/QXAI-prj/blob/main/Contingency_Matrix.png" width="500" /> 

&emsp; The chi-squared value can be calculated using the following formula.

$$\chi^2 = \frac{(|b-c|-1)^2}{b+c} = \frac{11^2}{40} = 3.025$$

$$df = 1$$

$$\alpha = 0.10$$

$$p-value = P(\chi^2 > 3.025) = 0.082 < 0.10$$

The QXAI model received an accuracy of 86.52% while the XAI model received an accuracy of 89.46%, providing some evidence that the XAI model is better than the QXAI model. McNemar's test with correction was employed to determine whether there was convincing evidence of one model performing better than the other. Because a p-value of 0.082 < 0.10, there is convincing evidence that one model is better than the other, and therefore, there is convincing evidence that the QXAI model performed worse than the XAI model. 
# How to Run
1. Clone the repository with "git clone https://github.com/jasirvatham2602/QXAI-prj" on the terminal, or simply download the contents of the repository as a zip file and extract it on Windows.
2. Open the QXAI-prj folder on the terminal/command prompt with "cd QXAI-prj"
3. The data has been compressed into a zip folder for convenience. Simply unzip it with "unzip data.zip" on the terminal, or right-click on the data file and unzip it on Windows
4. Now that the data file has been unzipped, we must install the needed dependencies with "pip install -r requirements.txt" or "pip3 install -r requirements.txt", if pip doesn't work.
5. Now run the code with "python main.py" or "python3 main.py" (Make sure Python is already installed on the computer. If it is not, you can download it at python.org)
# Limitation and Future work
Since access to a quantum computer was not available, simulation of qubits for the quantum circuit was done by PennyLane-- a common framework used in Quantum Computing. Access to a Quantum Computer may have increased the performance of the model. 
In the future, other Quantum Explainable AI techniques can be implemented to further help create trust between Quantum AI models and neurologists. Moreover, the classification of other neurological disorders can be done in the future with QXAI technology to help provide quick and precise diagnoses to patients with other neurological diseases.  
# Acknowledgements 
I would like to acknowledge all of the dependencies used in this passion project, such as PennyLane, PyTorch, Qiskit, matplotlib, seaborn, numpy, and more, as seen in the requirements.txt file. Pennylane was used to design the Variational Quantum Circuit of the QXAI architecture, and PyTorch was used for the classical AI parts. Other dependencies, such as numpy, matplotlib, and seaborn, were also used. Additionally, Qiskit was used to make the VQC diagram to help explain the QXAI architecture.
Furthermore, I would also like to acknowledge the Kaggle dataset, which provided the MRI scans of AD, PD, and CONTROL below. Some MRI scans were removed due to noise.
https://www.kaggle.com/datasets/farjanakabirsamanta/alzheimer-diseases-3-class 
