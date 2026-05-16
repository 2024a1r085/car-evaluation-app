# Car Evaluation AI

A Flask web app that uses a **Decision Tree classifier** to evaluate a car's
overall rating based on 6 key features — buying price, maintenance cost,
number of doors, passenger capacity, luggage boot size, and safety rating.

Trained on the [UCI Car Evaluation Dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation).

## 🔍 How It Works

Select the car's features and the model instantly predicts one of four ratings:

| Rating | Meaning |
|---|---|
| 🔴 UNACCEPTABLE | Not worth considering |
| 🟡 ACCEPTABLE | Meets minimum standards |
| 🟢 GOOD | A solid choice |
| 🔵 VERY GOOD | Highly recommended |

## 🚀 Getting Started

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the model
python train_model.py

### 3. Run the app
python app.py

Then open **http://localhost:5000** in your browser.

## 🛠️ Tech Stack
- Python, Flask
- scikit-learn (Decision Tree Classifier)
- HTML/CSS (Jinja2 templates)
- UCI Car Evaluation Dataset
