# 🧾 Invoice Intelligence ML

An end-to-end **Machine Learning and Streamlit application** for intelligent invoice analysis. The system uses two ML models to help businesses estimate freight costs and identify potentially suspicious invoices.

## 🚀 Features

### 🚚 Freight Cost Prediction

Predicts the estimated freight cost using:

* Invoice Dollars
* Invoice Quantity

**Input:**

```text
Dollars + Quantity
        ↓
Freight Cost Prediction
```

### 🚩 Invoice Flagging & Anomaly Detection

Classifies invoices as potentially flagged or normal using:

* `invoice_quantity`
* `invoice_dollars`
* `Freight`
* `total_brands`
* `total_item_quantity`
* `days_po_to_invoice`
* `total_item_dollars`

**Input:**

```text
Invoice Features
       ↓
Classification Model
       ↓
Flag / Normal
```

### 📊 Dashboard

The Streamlit frontend provides:

* Overview & Analytics
* Freight Cost Predictor
* Invoice Flagging & Anomaly Detection
* SQLite Database Viewer
* Model status monitoring
* Prediction results
* Model input inspection
* CSV export for database records

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Pickle**
* **SQLite**
* **Streamlit**
* **Matplotlib**
* **Git & GitHub**

---

## 📁 Project Structure

```text
Invoice-Intelligence-ML/
│
├── Dataset/
│   └── inventory.db
│
├── Models/
│   ├── Best_freight_model.pkl
│   └── Flagging Vendor Invoices.pkl
│
├── notebooks/
│   ├── Freight_Model.ipynb
│   └── Invoice_Flagging_Model.ipynb
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

> File and folder names may vary depending on the final project structure.

---

## 🤖 Machine Learning Models

### 1. Freight Cost Regression

The regression model predicts the freight cost based on invoice information.

### Features

```python
X = df[['Quantity', 'Dollars']]
```

### Target

```python
y = df['Freight']
```

The best-performing regression model is saved as:

```text
Best_freight_model.pkl
```

---

### 2. Invoice Classification

The classification model predicts whether an invoice should be flagged.

### Features

```python
X = df[
    [
        'invoice_quantity',
        'invoice_dollars',
        'Freight',
        'total_brands',
        'total_item_quantity',
        'days_po_to_invoice',
        'total_item_dollars'
    ]
]
```

### Target

```python
y = df['flag_invoice']
```

The trained model is saved as:

```text
Flagging Vendor Invoices.pkl
```

---

## 🗄️ Database

The project uses **SQLite** to store and explore invoice-related data.

Database:

```text
Dataset/inventory.db
```

The Streamlit application automatically detects available tables and allows users to:

* Select a database table
* View invoice records
* Control the number of displayed rows
* Download displayed data as CSV

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/ashirbutt/Invoice-Intelligence-ML.git
```

Move into the project directory:

```bash
cd Invoice-Intelligence-ML
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run app/app.py
```

The application will open in your browser.

Usually:

```text
http://localhost:8501
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
```

SQLite is included with Python, so a separate SQLite installation is generally not required for this application.

---

## 📈 Application Workflow

```text
                 ┌─────────────────────┐
                 │   Invoice Database  │
                 │     inventory.db    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Data Preprocessing │
                 └──────────┬──────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
       ┌─────────────────┐     ┌──────────────────┐
       │ Freight Model   │     │ Invoice Flagging │
       │   Regression    │     │ Classification   │
       └────────┬────────┘     └─────────┬────────┘
                │                        │
                ▼                        ▼
       ┌─────────────────┐     ┌──────────────────┐
       │ Estimated       │     │ Flag / Normal    │
       │ Freight Cost    │     │ Invoice Result   │
       └────────┬────────┘     └─────────┬────────┘
                │                        │
                └───────────┬────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit Dashboard │
                 └─────────────────────┘
```

---

## 🎯 Project Objective

The objective of **Invoice Intelligence ML** is to demonstrate how Machine Learning can be integrated into an invoice management workflow.

The system focuses on two practical business problems:

1. **Freight Cost Prediction**
   Estimate expected freight costs from invoice information.

2. **Invoice Flagging**
   Identify invoices that may require additional review.

The project combines **Machine Learning, data processing, SQLite, model serialization, and Streamlit** into a single interactive application.

---

## 🔮 Future Improvements

Possible future enhancements include:

* Automated invoice data ingestion
* PDF invoice processing
* OCR-based invoice extraction
* Real-time database integration
* Advanced anomaly detection
* Model performance monitoring
* Authentication and user management
* REST API using FastAPI
* Docker deployment
* Cloud deployment
* Automated model retraining
* Interactive analytics and charts

---

## 👨‍💻 Author

**Muhammad Ashir**

Machine Learning & AI Engineer

---

## 📄 License

This project is intended for educational, portfolio, and demonstration purposes.
