# 🧠 Receipt_Count Prediction App

This project predicts the number of scanned receipts for each month of 2022 using daily receipt data from 2021.  
The ML model is implemented from scratch using NumPy and wrapped inside an interactive Flask web app.

---

## 🚀 Features

- Polynomial regression model
- Inference with dynamic month selection
- Side-by-side comparison with 2021 actual data
- Visual forecast using dual charts
- Fully Dockerized and easy to run

---

## 🧠 Tech Stack

- Python
- NumPy
- Flask
- Matplotlib
- Docker

---

## 🛠 How to Run

### 🔹 Option 1: Local Python

1. Install dependencies:

   ```bash
   pip install -r requirements.txt

2. Run the app:

   ```bash
   python app.py

3. Open your browser:

   http://localhost:5000  

### 🔹 Option 2:  Docker

1. Make sure Docker is installed and running

2. Build the image:

    ```bash
    docker build -t receipt-forecast .

3. Run the container:

    ```bash
    docker run -p 5000:5000 receipt-forecast

4. Open your browser:

   http://localhost:5000  

### OR

### Public Docker Image 

1. You can also skip building and just pull the public image:

    ```bash
    docker run -p 5000:5000 menonabhineet/receipt-forecast