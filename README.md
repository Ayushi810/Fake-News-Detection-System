# 📰 Fake News Detection System

A simple Machine Learning web application that classifies news text as **REAL** or **FAKE**.

## 🚀 Features

- TF-IDF based text feature extraction
- Logistic Regression classifier
- Streamlit web interface
- Confidence score
- Ready-to-run trained model
- Easy model retraining using `train_model.py`

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── app.py
├── train_model.py
├── fake_news.csv
├── requirements.txt
├── README.md
└── model/
    ├── fake_news_model.pkl
    └── tfidf_vectorizer.pkl
```

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit app

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧠 How It Works

1. User enters a news headline or article.
2. Text is converted into numerical features using **TF-IDF**.
3. A **Logistic Regression** model predicts the class.
4. The application displays REAL/FAKE prediction and model confidence.

## 🔄 Retrain the Model

To train the model again after changing `fake_news.csv`:

```bash
python train_model.py
```

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. It is not a replacement for professional fact-checking or trusted news sources.

## 👩‍💻 Author

**Ayushi Kumari**
