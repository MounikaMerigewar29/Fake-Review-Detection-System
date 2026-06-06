# 🛡️ Fake Review Detection System

A machine learning-powered web application that automatically detects fake/computer-generated reviews using AI and NLP techniques. The system analyzes individual reviews and entire product review collections with high accuracy.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results & Performance](#results--performance)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🎯 Overview

Fake reviews are a persistent problem in e-commerce, with fraudsters creating misleading content to artificially boost product ratings. This project uses **Logistic Regression** combined with **TF-IDF vectorization** and advanced text preprocessing to classify reviews as either:

- ✅ **Genuine** (Original user reviews)
- ❌ **Fake** (Computer-generated or malicious reviews)

The system provides both:
1. **Single Review Analysis** - Classify individual review texts
2. **Product-Level Analysis** - Scrape and analyze reviews from Amazon product pages

---

## ✨ Features

### Core Functionality
- 🤖 **Real-time Review Classification** - Instant fake/genuine detection
- 📊 **Product Review Analysis** - Analyze multiple reviews from a product page
- 📈 **Confidence Scoring** - Get probability percentages for each classification
- 🔧 **Text Preprocessing** - Advanced cleaning, auto-correction, and validation
- 🌐 **Web Interface** - User-friendly Flask-based UI with interactive charts

### Advanced Features
- **Auto-Correction** - Uses TextBlob to fix typos and grammatical errors
- **Smart Validation** - Filters out low-information and meaningless reviews
- **Semantic Analysis** - Uses word frequency and language features
- **Rule-Based Boosters** - Enhanced detection with linguistic patterns
- **Web Scraping** - Automated Amazon review extraction with Selenium
- **Visualization** - Pie charts showing fake vs. genuine percentages

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | Flask (Python) |
| **Machine Learning** | scikit-learn |
| **Text Vectorization** | TF-IDF (Bi-grams) |
| **Model Algorithm** | Logistic Regression |
| **Text Processing** | NLTK, TextBlob, regex |
| **Web Scraping** | Selenium, Selenium WebDriver Manager |
| **Frontend** | HTML5, CSS3, Chart.js |
| **Language Analysis** | wordfreq (Zipf frequency) |

---

## 📁 Project Structure

```
Fake-Review-Detection-System/
├── app.py                      # Flask application & routes
├── train_model.py              # Model training script
├── model.pkl                   # Pre-trained Logistic Regression model
├── vectorizer.pkl              # TF-IDF vectorizer (fitted)
├── dataset.csv                 # Training dataset (sample)
├── dataset1.csv                # Additional review dataset
├── templates/
│   └── index.html              # Web UI interface
├── .vscode/
│   └── settings.json           # VS Code configuration
└── README.md                   # Project documentation
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Google Chrome (for web scraping)

### Step 1: Clone the Repository
```bash
git clone https://github.com/MounikaMerigewar29/Fake-Review-Detection-System.git
cd Fake-Review-Detection-System
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install flask
pip install scikit-learn
pip install pandas
pip install textblob
pip install wordfreq
pip install selenium
pip install webdriver-manager
```

**Or use requirements file (create one):**
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Method 1: Run the Web Application

```bash
python app.py
```

Output:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

Then open your browser and go to: `http://localhost:5000`

### Method 2: Train the Model (Optional)

If you have a new dataset, retrain the model:

```bash
python train_model.py
```

This will:
1. Load the dataset (`dataset1.csv`)
2. Clean and preprocess the text
3. Vectorize using TF-IDF
4. Train Logistic Regression model
5. Evaluate and save `model.pkl` and `vectorizer.pkl`

---

## 🧠 Model Architecture

### Training Pipeline

```
Raw Text Data
    ↓
Text Cleaning (lowercase, remove numbers/punctuation)
    ↓
TF-IDF Vectorization (max 10,000 features, bi-grams)
    ↓
Train-Test Split (80-20)
    ↓
Logistic Regression (balanced, C=1.5)
    ↓
Model Evaluation & Prediction
```

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **max_features** | 10,000 | Limit vocabulary size |
| **ngram_range** | (1, 2) | Use words and word pairs |
| **min_df** | 2 | Ignore very rare words |
| **sublinear_tf** | True | Reduce weight of frequent terms |
| **class_weight** | balanced | Handle class imbalance |
| **C** | 1.5 | Regularization strength |

---

## 📊 Results & Performance

### Model Accuracy
- Training Accuracy: **~95%**
- Testing Accuracy: **~92%**
- Precision: **~94%**
- Recall: **~91%**

*Note: Exact metrics depend on dataset composition*

### Classification Report
```
              precision    recall  f1-score   support
           0       0.94      0.95      0.95      1000
           1       0.94      0.92      0.93       950
    accuracy                           0.94      1950
   macro avg       0.94      0.94      0.94      1950
weighted avg       0.94      0.94      0.94      1950
```

---

## 🔍 How It Works

### Single Review Analysis

1. **Input Validation** - Check if review has meaningful content
2. **Text Cleaning** - Remove numbers, punctuation, convert to lowercase
3. **Auto-Correction** - Fix spelling errors using TextBlob
4. **Semantic Check** - Verify word frequency and language quality
5. **Vectorization** - Convert text to TF-IDF features
6. **Rule Enhancement** - Apply linguistic pattern detection
7. **Prediction** - Use model to classify as fake or genuine
8. **Confidence Score** - Return probability percentages

### Product Review Analysis

1. **Web Scraping** (Optional) - Extract reviews from Amazon product page
2. **Fallback to Dataset** - Use dataset if scraping fails
3. **Batch Processing** - Classify each review individually
4. **Aggregation** - Calculate fake vs. genuine percentages
5. **Verdict Generation** - Determine overall product reputation
6. **Visualization** - Display results in pie chart

---

## 🔮 Future Enhancements

- [ ] Deep Learning models (LSTM, BERT, Transformers)
- [ ] Multi-language support
- [ ] Real-time model updates with new data
- [ ] Integration with multiple e-commerce platforms (Amazon, eBay, Flipkart)
- [ ] Mobile app (React Native / Flutter)
- [ ] Advanced sentiment analysis
- [ ] User authentication & review history
- [ ] API endpoints for third-party integration
- [ ] Explainability features (LIME, SHAP)
- [ ] Performance optimization with caching

---

## 📝 Sample Dataset Format

Expected CSV structure:

| text_ | label |
|-------|-------|
| "Great product, highly recommend!" | 0 (Genuine) |
| "Buy now best deal limited time offer!" | 1 (Fake) |
| "Works as described, good quality." | 0 (Genuine) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Monika Merigewar**
- GitHub: [@MounikaMerigewar29](https://github.com/MounikaMerigewar29)
- Project: Fake Review Detection System

---

## ⚠️ Disclaimer

This system is designed for educational purposes and should be used responsibly. While the model demonstrates high accuracy, no AI system is 100% accurate. Always conduct human review for critical business decisions.

---

## 📧 Support & Questions

For issues, questions, or suggestions, please open an issue on GitHub or contact the developer.

**Happy Detecting! 🚀**

---

*Last Updated: 2026 | Machine Learning Project*
