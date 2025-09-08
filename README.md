# ECE 4424 / CS 4824 

A comprehensive, hands-on machine learning course using interactive Jupyter notebooks. Build real ML systems while understanding the fundamental principles.

## Course Philosophy

Inspired by fast.ai, our motto is "build first, understand later." We believe the best way to develop a deep intuition for machine learning is to start by building real systems.

**Note on AI in Development:** To model modern and transparent development practices, AI-powered tools were used to assist in the creation of these course materials. All content has been directed, reviewed, and validated by the instructor.

## Course Overview

### What You'll Learn
- **Foundations**: Core ML algorithms (kNN, Linear/Logistic Regression, Trees)
- **Deep Learning**: Neural Networks, CNNs, RNNs, Transformers
- **Advanced Topics**: Reinforcement Learning, Generative AI, AI Safety
- **Practical Skills**: Data handling, model evaluation, deployment

### Prerequisites
- Basic Python programming
- High school mathematics
- Curiosity and willingness to experiment

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/practical-ml-course.git
cd practical-ml-course
```

### 2. Create a Virtual Environment
```bash
python3 -m venv ml_lectures_env
source ml_lectures_env/bin/activate   # Windows: ml_lectures_env\\Scripts\\activate
```

### 3. Install Requirements
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Note: The requirements include deep learning dependencies `torch==2.2.2` and `torchvision==0.17.2` (CPU wheels). First install can take several minutes and a few hundred MB of disk.

### 4. Launch Jupyter Notebook (classic)
```bash
jupyter notebook
```

Optional: Register the environment as a named kernel to select it in notebooks:
```bash
python -m ipykernel install --user --name ml_lectures_env --display-name "Python (ml_lectures_env)"