# import re
# import pickle
# from flask import Flask, render_template, request
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# import time

# app = Flask(__name__)

# # Load model and vectorizer
# model = pickle.load(open("model.pkl", "rb"))
# vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# # Cleaning function (same as training)

# def get_reviews_from_amazon(url):
#     options = webdriver.ChromeOptions()
#     options.add_argument("--headless")
#     options.add_argument("--disable-blink-features=AutomationControlled")

#     driver = webdriver.Chrome(options=options)
#     driver.get(url)

#     time.sleep(5)

#     reviews = []

#     try:
#         elements = driver.find_elements(By.XPATH, "//span[@data-hook='review-body']")

#         for el in elements:
#             reviews.append(el.text)

#     except Exception as e:
#         print("Error:", e)

#     driver.quit()

#     return reviews

# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'\d+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     review = request.form['review']
#     cleaned_review = clean_text(review)

#     # Short input detection
#     if len(cleaned_review.split()) < 3:
#         result = "Low Confidence: Review too short for reliable classification."
#         confidence = 0
#         return render_template('index.html', prediction=result, confidence=confidence)

#     vectorized_review = vectorizer.transform([cleaned_review])

#     prediction = model.predict(vectorized_review)[0]
#     probabilities = model.predict_proba(vectorized_review)[0]

#     # Get correct class order
#     class_labels = model.classes_
#     prob_dict = dict(zip(class_labels, probabilities))

#     # Extract probabilities safely
#     genuine_prob = round(prob_dict['OR'] * 100, 2)
#     fake_prob = round(prob_dict['CG'] * 100, 2)

#     if prediction == 'CG':
#         result = "Computer Generated / Fake Review"
#         confidence = fake_prob
#     else:
#         result = "Original / Genuine Review"
#         confidence = genuine_prob

#     # Low confidence warning
#     if confidence < 60:
#         result += " (Low Confidence)"

#     return render_template(
#         'index.html',
#         prediction=result,
#         confidence=confidence,
#         fake_probability=fake_prob,
#         genuine_probability=genuine_prob
#     )

# @app.route('/analyze_link', methods=['POST'])
# def analyze_link():
#     import pandas as pd

#     link = request.form['link']

#     df = pd.read_csv("dataset1.csv")
#     reviews = df['text_'][:100]

#     fake = 0
#     genuine = 0

#     for review in reviews:
#         cleaned = clean_text(review)
#         vec = vectorizer.transform([cleaned])
#         pred = model.predict(vec)[0]

#         if pred == 1:
#             fake += 1
#         else:
#             genuine += 1

#     total = fake + genuine

#     fake_percent = round((fake / total) * 100, 2)
#     genuine_percent = round((genuine / total) * 100, 2)

#     return render_template(
#         'index.html',
#         fake_percent=fake_percent,
#         genuine_percent=genuine_percent
#     )

# @app.route('/test')
# @app.route('/test')
# def test():
#     link = "https://www.amazon.in/product-reviews/B08N5WRWNW"
#     reviews = get_reviews_from_amazon(link)

#     return f"Fetched {len(reviews)} reviews"
    
# if __name__ == '__main__':
#     app.run(debug=True)




import re
import pickle
from flask import Flask, render_template, request
from textblob import TextBlob
from wordfreq import zipf_frequency

app = Flask(__name__)

# -------- LOAD MODEL --------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------- CLEAN TEXT --------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------- VALIDATION --------
def is_meaningful_text(text):
    words = text.split()

    if len(words) < 2:
        return False

    meaningful_count = 0

    for word in words:
        freq = zipf_frequency(word, 'en')

        if freq > 2 and any(c in "aeiou" for c in word):
            meaningful_count += 1

    return (meaningful_count / len(words)) >= 0.4

# -------- LOW INFO --------
def is_low_information(text):
    return len(text.split()) <= 4

# -------- AUTO CORRECT --------
def autocorrect_text(text):
    try:
        return str(TextBlob(text).correct())
    except:
        return text

# -------- SCRAPER --------
def get_reviews_from_amazon(url):
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        import time
        import random

        options = Options()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("start-maximized")
        options.add_argument("user-agent=Mozilla/5.0")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

        driver.get(url)
        time.sleep(random.uniform(4, 7))

        reviews = []

        elements = driver.find_elements(By.XPATH, "//span[@data-hook='review-body']")
        for el in elements:
            reviews.append(el.text.strip())

        driver.quit()
        return reviews

    except Exception as e:
        print("Scraping error:", e)
        return []

# -------- HOME --------
@app.route('/')
def home():
    return render_template('index.html')

# -------- SINGLE REVIEW --------
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')

    cleaned = clean_text(review)

    # Invalid input
    if not is_meaningful_text(cleaned):
        return render_template(
            'index.html',
            prediction="Invalid Input",
            confidence=0,
            fake_probability=0,
            genuine_probability=0
        )

    # Auto-correct
    corrected = autocorrect_text(review)
    cleaned = clean_text(corrected)

    # Low info
    if is_low_information(cleaned):
        return render_template(
            'index.html',
            prediction="Low Information Review",
            confidence=50,
            fake_probability=50,
            genuine_probability=50,
            corrected_text=corrected
        )

    # Prediction
    # vector = vectorizer.transform([cleaned])
    # pred = model.predict(vector)[0]
    # probs = model.predict_proba(vector)[0]

    # prob_dict = dict(zip(model.classes_, probs))

    # fake_prob = round(prob_dict.get('CG', 0) * 100, 2)
    # genuine_prob = round(prob_dict.get('OR', 0) * 100, 2)

    # # if pred == 'CG':
    # #     result = "Computer Generated / Fake Review"
    # #     confidence = fake_prob
    # # else:
    # #     result = "Original / Genuine Review"
    # #     confidence = genuine_prob
    # if fake_prob > 30:   # you can tune (35–45)
    #     result = "Computer Generated / Fake Review"
    #     confidence = fake_prob
    # else:
    #     result = "Original / Genuine Review"
    #     confidence = genuine_prob

    # if confidence < 60:
    #     result += " (Low Confidence)"
    vector = vectorizer.transform([cleaned])
    probs = model.predict_proba(vector)[0]

    prob_dict = dict(zip(model.classes_, probs))

    fake_prob = round(prob_dict.get('CG', 0) * 100, 2)
    genuine_prob = round(prob_dict.get('OR', 0) * 100, 2)

    # 🔥 RULE BOOST
    if cleaned.count("!") >= 2 or "buy buy" in cleaned:
        fake_prob += 80

    # 🔥 FINAL DECISION
    if fake_prob > 30:
        result = "Computer Generated / Fake Review"
        confidence = fake_prob
    else:
        result = "Original / Genuine Review"
        confidence = genuine_prob

    if confidence < 60:
        result += "\nNOT RELATED (Low Confidence)"

    return render_template(
        'index.html',
        prediction=result,
        confidence=confidence,
        fake_probability=fake_prob,
        genuine_probability=genuine_prob,
        corrected_text=corrected
    )

# -------- LINK ANALYSIS --------
@app.route('/analyze_link', methods=['POST'])
def analyze_link():
    url = request.form.get('product_link')

    if not url:
        return "Error: No link received"

    reviews = get_reviews_from_amazon(url)

    # Fallback if scraping fails
    if len(reviews) == 0:
        import pandas as pd
        df = pd.read_csv("dataset1.csv")
        reviews = df['text_'][:50]
        source = "Dataset"
    else:
        source = "Live"

    fake = 0
    genuine = 0

    for review in reviews:
        cleaned = clean_text(review)

        if not is_meaningful_text(cleaned):
            continue

        corrected = autocorrect_text(review)
        cleaned = clean_text(corrected)

        if is_low_information(cleaned):
            continue

        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)[0]

        if pred == 'CG':
            fake += 1
        else:
            genuine += 1

    total = fake + genuine

    if total == 0:
        fake_percent = 0
        genuine_percent = 0
    else:
        fake_percent = round((fake / total) * 100, 2)
        genuine_percent = round((genuine / total) * 100, 2)

    # Verdict
    if fake_percent > 60:
        verdict = "Mostly Fake Reviews"
    elif genuine_percent > 60:
        verdict = "Mostly Genuine Reviews"
    else:
        verdict = "Mixed Reviews"

    return render_template(
        'index.html',
        fake_percent=fake_percent,
        genuine_percent=genuine_percent,
        verdict=verdict,
        source=source
    )

# -------- RUN --------
if __name__ == '__main__':
    app.run(debug=True)