# **🔥 PyTorch Learning Journey**

> *"The only way to learn mathematics is to do mathematics."* — Paul Halmos  
> *So here I am, doing the math... with tensors.*

---

## **🧠 What's This?**

My personal exploration through the depths of **Machine Learning**, one **Gradient Descent** at a time. This repository chronicles my journey through neural networks, backpropagation, and the beautiful chaos of training models.

Currently powered by **PyTorch** 🔥 — because who doesn't love dynamic computational graphs?

---

## **📚 The Curriculum**

I'm simultaneously diving into:
- **Machine Learning** → Neural Networks & Deep Learning
- **Data Science** → Making sense of chaos
- **Calculus** → The language of change
- **Probability Theory** → Embracing uncertainty
- **Statistics** → Numbers that tell stories

---

## 🗂️ Repository Structure
```
pytorch/
├── notebooks/          # The brain of the operation
│   ├── notebook1.ipynb      # 2-layer NN with BCE loss
│   ├── notebook2.ipynb      # Scaled network with MSE loss
│   └── notebook3.ipynb      # L2 regularization experiment
├── README.md                # You are here
├── requirements.txt         # Dependencies
└── .gitignore              # What stays hidden
```

---

## **🎯 Core Concepts & Methods**

### **1️⃣ Neural Network Fundamentals**

#### **Architecture Building Blocks:**
- **Layers:** Input → Hidden → Output
- **Activations:** ReLU (hidden), Sigmoid (output)
- **Parameters:** Weights and biases learned during training

#### **What I'm Learning:**
- How neurons combine inputs with weights
- Why non-linear activations (ReLU) are crucial
- How to design layer sizes for different tasks

### **2️⃣ Training Process**

#### **Core Components:**
- **Forward Pass:** Data flows through network → predictions
- **Loss Calculation:** How wrong are we?
- **Backpropagation:** Calculate gradients (which way to improve)
- **Optimization:** Update weights to reduce error

#### **Optimization Method:**
- **Adam Optimizer:** Adaptive learning rates for each parameter
- **Learning Rate:** 0.001 (controls update step size)

### **3️⃣ Loss Functions Explored**

#### **Binary Cross-Entropy (BCE):**
```math
Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
- Designed for classification
- Outputs true probabilities
- Used in: Notebook 1

#### **Mean Squared Error (MSE):**
```math
Loss = (y - ŷ)²
```
- Traditionally for regression
- Simpler gradients
- Used in: Notebooks 2 & 3

### **4️⃣ Regularization Techniques**

#### **L2 Weight Decay:**
- Adds penalty for large weights: `λ × Σ(weights²)`
- Prevents overfitting by constraining model complexity
- Implemented via optimizer: `weight_decay=0.01`

#### **Purpose:**
- Prevent memorization of training data
- Improve generalization to new data
- Create smoother decision boundaries

### **5️⃣ Data Strategy**

#### **Synthetic Data Generation:**
- Random features with hidden non-linear patterns
- 5% label noise to simulate real-world errors
- Split: 80% training, 20% validation

#### **Why This Approach?**
- Tests if network can discover patterns in chaos
- Validates learning on truly unseen data
- Builds robust models that generalize

### **6️⃣ Diagnostic Tools**

#### **Loss Curves:**
- Track training and validation loss over epochs
- Identify overfitting (train↓, val↑)
- Identify underfitting (both stay high)

#### **What Healthy Training Looks Like:**
```
Both curves decreasing together ✅
Small gap between curves ✅
Smooth convergence ✅
```

---

## **🛠️ Tech Stack**
```python
framework = {
    "deep_learning": "PyTorch",
    "language": "Python 3.x",
    "computation": "NumPy",
    "visualization": "Matplotlib",
    "data_processing": "scikit-learn",
    "environment": "Jupyter Notebook"
}
```

---

## **📊 Project Progress**

### **Completed Experiments:**

| Notebook | Architecture | Loss Function | Regularization | Status |
|----------|-------------|---------------|----------------|---------|
| 1 | 2→4→1 (17 params) | BCE | None | ✅ Complete |
| 2 | 3→5→1 (26 params) | MSE | None | ✅ Complete |
| 3 | 4→7→1 (43 params) | MSE | L2 (0.01) | ✅ Complete |

### **Key Experiments Conducted:**

#### **Experiment 1: Baseline Neural Network**
- Established performance baseline
- Observed mild overfitting
- Final loss: 0.62

#### **Experiment 2: Scaling Up (Data + Model)**
- Increased dataset 50% (1000→1500 samples)
- Increased parameters 53% (17→26)
- Switched to MSE loss
- Result: 71% improvement (loss: 0.17)

#### **Experiment 3: L2 Regularization**
- Added weight decay to prevent overfitting
- Most parameters yet (43)
- Perfect train/val alignment
- Trade-off: Higher loss (0.67) for stability

---

## **💡 Key Learnings So Far**

### **🎓 Lesson 1: Loss Function Matters**
> **MSE outperformed BCE** in our case (0.17 vs 0.62). But context matters - MSE worked better with more data and simpler gradients.

### **🎓 Lesson 2: Data > Model Size**
> **More data beats bigger models.** Going from 800→1200 samples had more impact than architectural changes.

### **🎓 Lesson 3: Regularization is Medicine**
> **Don't use it if you're not sick.** L2 prevented overfitting but hurt performance when overfitting wasn't present.

### **🎓 Lesson 4: Loss Curves Tell Stories**
> **Train and validation curves are your best diagnostic tool.** They reveal overfitting, underfitting, and convergence issues instantly.

---

## **📈 Visual Progress**

### **Performance Evolution:**
```
Notebook 1 (BCE, No Reg):     Loss = 0.62
Notebook 2 (MSE, No Reg):     Loss = 0.17  ⭐ Best
Notebook 3 (MSE, L2 Reg):     Loss = 0.67
```

### **What This Shows:**
- Proper architecture + loss function > regularization
- Sometimes simpler is better
- Regularization should be used strategically

---

## **🔮 Future Plans**

### **Next Experiments:**
- [ ] Dropout regularization comparison
- [ ] Deeper networks (3+ layers)
- [ ] Different activation functions (Tanh, LeakyReLU)
- [ ] Real-world datasets
- [ ] Transfer to TensorFlow/JAX for comparison

---

## **🎯 Philosophy**

This isn't just about completing homework. It's about:
- **Understanding** over memorization
- **Building** intuition through implementation
- **Visualizing** abstract concepts
- **Experimenting** fearlessly

>Every experiment is an opportunity. Every error is a lesson. Every gradient brings me closer to understanding.

---

## **📚 Core Concepts Reference**

### **Backpropagation in Simple Terms:**
1. Make prediction (forward pass)
2. Calculate error (loss function)
3. Compute gradients (how to improve)
4. Update weights (take a step)
5. Repeat until convergence

### **The Bias-Variance Tradeoff:**
- **High Bias:** Model too simple (underfitting)
- **High Variance:** Model too complex (overfitting)
- **Goal:** Sweet spot in the middle

### **Overfitting vs Underfitting:**
- **Overfitting:** Train loss ↓↓, Val loss ↑ (memorizing)
- **Underfitting:** Both losses high (not learning)
- **Good Fit:** Both losses low and close (generalizing)

---

## **📅 Timeline**

**Last Updated:** *November 18, 2025*

**Project Started:** *November 16, 2025*

**Notebooks Completed:** 3/3 (Basic Series)

---

## **⚡ Fun Facts**

- The number of parameters in GPT-3 (175 billion) is roughly equal to the number of synapses in a human brain region called the cerebellum
- Backpropagation was rediscovered multiple times before becoming the standard training method
- Adam optimizer combines momentum and adaptive learning rates - best of both worlds!

---

<div align="center">

**Built with 🔥 PyTorch and ☕ Coffee**

*"In God we trust, all others must bring data."* — W. Edwards Deming

</div>