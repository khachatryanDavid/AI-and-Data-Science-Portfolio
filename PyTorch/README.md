# 🔥 **PyTorch Deep Learning Journey**

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com/khachatryanDavid/AI-and-Data-Science-Portfolio)
[![Notebooks](https://img.shields.io/badge/Notebooks-3%2F3-blue?style=for-the-badge)](./notebooks/)

*Building artificial minds, one gradient descent at a time*

> *"The only way to learn mathematics is to do mathematics."* — Paul Halmos  
> *So here I am, doing the math... with tensors.*

</div>

---

## 🧠 **What's This?**

My personal exploration through the depths of **Deep Learning**, documenting the journey from basic neural networks to advanced training techniques. This repository demonstrates **practical understanding** of how neural networks learn, generalize, and optimize.

**Focus:** Not just running code — **understanding the why** behind every parameter, every activation, every gradient step.

---

## 📚 **The Curriculum**

Building neural network intuition through parallel study:

| Domain | Purpose | Application |
|--------|---------|-------------|
| **Machine Learning** | Neural Networks & Deep Learning | PyTorch implementation |
| **Calculus** | Derivatives, Chain Rule, Optimization | Backpropagation mechanics |
| **Probability Theory** | Distributions, Uncertainty | Loss functions, generalization |
| **Statistics** | Hypothesis testing, Variance | Model evaluation, diagnostics |
| **Data Science** | Visualization, Analysis | Training dynamics understanding |

---

## 🗂️ **Repository Structure**
```
pytorch/
├── notebooks/                   # Neural network experiments
│   ├── notebook1.ipynb         # Baseline architecture exploration
│   ├── notebook2.ipynb         # Scaling strategies and loss functions
│   └── notebook3.ipynb         # Regularization techniques
└── README.md                   # This file
```

---

## 🎯 **Core Concepts & Methods**

### **Neural Network Fundamentals**

**Architecture Building Blocks:**
- **Layers:** Sequential transformations (Input → Hidden → Output)
- **Activations:** Non-linear functions (ReLU, Sigmoid, Tanh)
- **Parameters:** Learnable weights and biases
- **Forward Pass:** Data flow through network
- **Backpropagation:** Gradient computation for learning

**Key Understanding:**  
Neural networks are **universal function approximators** — they learn patterns by adjusting millions of parameters through gradient descent.

---

### **Training Process**

**The Learning Pipeline:**
```
Data → Forward Pass → Loss Calculation → Backpropagation → 
Weight Update → Repeat Until Convergence
```

**Optimization:**
- **Algorithm:** Adam (adaptive learning rates)
- **Learning Rate:** 0.001 (controls step size)
- **Epochs:** Multiple passes through dataset
- **Goal:** Minimize loss function

---

### **Loss Functions**

**Binary Cross-Entropy (BCE):**
- Designed for classification tasks
- Outputs calibrated probabilities
- Used when you need true probability estimates

**Mean Squared Error (MSE):**
- Simpler gradient behavior
- Works well with sufficient data
- Used when prediction accuracy matters most

**Experimental Finding:** Loss function choice significantly impacts performance — MSE showed 71% improvement over BCE in scaled experiments.

---

### **Regularization Techniques**

**L2 Weight Decay:**
- Prevents overfitting by penalizing large weights
- Adds `λ × Σ(weights²)` to loss function
- Trade-off: Better generalization vs potential underfitting

**Key Insight:** Regularization is **medicine** — only apply when overfitting is present.

---

### **Data Strategy**

**Synthetic Data Generation:**
- Random features with hidden non-linear patterns
- 5% label noise (simulates real-world imperfection)
- 80/20 train/validation split
- Tests model's ability to discover structure in chaos

---

### **Diagnostic Tools**

**Loss Curves:**
- Primary tool for understanding training dynamics
- Identify overfitting, underfitting, and convergence
- Visualize relationship between train and validation loss

**Healthy Training Indicators:**
```
✅ Both losses decreasing together
✅ Small gap between train/val curves
✅ Smooth convergence
✅ Validation loss stabilizes
```

---

## 🛠️ **Tech Stack**
```python
framework = {
    "deep_learning": "PyTorch 2.0+",
    "language": "Python 3.8+",
    "computation": "NumPy",
    "visualization": "Matplotlib",
    "data_processing": "scikit-learn",
    "environment": "Jupyter Notebook"
}
```

**Why PyTorch?**
- Dynamic computational graphs (debug like Python)
- Research-friendly API
- Industry standard for deep learning
- Excellent documentation and community

---

## 📊 **Project Progress**

### **Completed Experiments:**

| Notebook | Focus Area | Key Concept | Status |
|----------|-----------|-------------|---------|
| **1** | Baseline Architecture | Network fundamentals, BCE loss | ✅ Complete |
| **2** | Scaling Strategies | Data + model scaling, MSE vs BCE | ✅ Complete |
| **3** | Regularization | L2 weight decay, bias-variance tradeoff | ✅ Complete |

### **Key Achievements:**

🏆 **71% Performance Improvement** through strategic architecture scaling  
🔬 **Zero Overfitting** achieved despite model complexity increase  
📊 **Perfect Convergence** (train/val gap: 0.0005) with L2 regularization  
🎓 **Deep Understanding** of loss functions, optimization, and diagnostics  

---

## 💡 **Key Learnings**

### **Lesson 1: Loss Function Impact**
> Different loss functions dramatically affect performance. MSE outperformed BCE by 71% in scaled experiments — but context matters.

### **Lesson 2: Data Trumps Architecture**
> More high-quality data beats complex architectures. Scaling from 800→1200 samples had more impact than doubling parameters.

### **Lesson 3: Regularization Strategy**
> Don't apply regularization blindly. L2 prevented overfitting but hurt performance when overfitting wasn't present. Diagnose first, then treat.

### **Lesson 4: Loss Curves as Diagnostics**
> Training and validation loss curves are the single best tool for understanding model behavior. They reveal overfitting, underfitting, and convergence instantly.

### **Lesson 5: Experimentation Over Theory**
> Hands-on experimentation builds intuition faster than theory alone. Every failed experiment teaches as much as successful ones.

---

## 📈 **Performance Summary**

### **Progression Across Notebooks:**
```
Notebook 1:  Loss = 0.62  →  Baseline established
Notebook 2:  Loss = 0.17  →  71% improvement ⭐ (Best performance)
Notebook 3:  Loss = 0.67  →  Perfect regularization, strategic trade-off
```

**What This Demonstrates:**
- Proper architecture + loss function > blind regularization
- Sometimes simpler approaches work best
- Understanding trade-offs is key to ML mastery

---

## 🔮 **Future Directions**

**Next Experiments:**
- [ ] Convolutional Neural Networks (CNNs)
- [ ] Recurrent architectures (RNNs, LSTMs)
- [ ] Attention mechanisms
- [ ] Dropout vs L2 regularization comparison
- [ ] Real-world datasets (MNIST, CIFAR-10)
- [ ] Transfer learning experiments

---

## 🎯 **Philosophy**

This project embodies:
- **Understanding over Memorization** — Know why, not just how
- **Experimentation over Perfection** — Learn from failures
- **Visualization over Abstraction** — Make the invisible visible
- **Rigor over Speed** — Deep learning requires deep understanding

> *"Every experiment is an opportunity. Every error is a lesson. Every gradient brings me closer to understanding."*

---

## 📚 **Core Principles**

### **Backpropagation Simplified:**
```
1. Forward pass: Make predictions
2. Calculate error: Loss function
3. Compute gradients: Chain rule
4. Update weights: Gradient descent
5. Repeat until convergence
```

### **The Bias-Variance Tradeoff:**
- **High Bias:** Model too simple (underfitting)
- **High Variance:** Model too complex (overfitting)
- **Goal:** Balance between the two

### **Training Diagnostics:**
- **Overfitting:** Train ↓↓, Val ↑ (memorizing)
- **Underfitting:** Both losses high (not learning)
- **Good Fit:** Both low and close (generalizing)

---

## 📅 **Timeline**

| Period | Milestone |
|--------|-----------|
| **November 16, 2025** | Project initiated |
| **November 17, 2025** | Notebook 1 completed (Baseline) |
| **November 18, 2025** | Notebooks 2-3 completed (Scaling & Regularization) |

---

## ⚡ **Did You Know?**

- **GPT-3** has 175 billion parameters — roughly equal to synapses in the cerebellum
- **Backpropagation** was discovered independently multiple times before becoming standard
- **Adam optimizer** combines momentum and adaptive learning — best of both worlds
- **ReLU activation** solved the vanishing gradient problem that plagued early neural networks
- **Dropout** was inspired by how biological neurons randomly fire

---

## 📬 **Connect**

Questions? Suggestions? Let's discuss deep learning!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/david-khachatryan-65a14b376/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github)](https://github.com/khachatryanDavid)

---

<div align="center">

### 💡 *"The brain is a three-pound mass you can hold in your hand that can conceive of a universe a hundred billion light-years across."*
*— Marian Diamond*

---

**🧠 Building artificial minds, one gradient descent at a time.**

---

📅 **Last Updated:** November 18, 2025  
👨‍💻 **Author:** David Khachatryan  
🔥 **Framework:** PyTorch 2.0+  
📊 **Notebooks:** 3/3 Complete  

---

**Built with curiosity, powered by tensors, driven by the pursuit of understanding intelligence.**

</div>
