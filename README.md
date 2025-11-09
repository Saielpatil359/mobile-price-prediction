# 📱 Mobile Price Range Prediction (ML + Streamlit App)

This project predicts the **price range** of a mobile device based on its specifications.  
Instead of predicting the exact price, it classifies phones into:

| Category | Meaning | Approx Price |
|--------|----------|--------------|
| 0 | 💰 Low Cost | ₹3,000 – ₹7,000 |
| 1 | 💵 Medium Cost | ₹8,000 – ₹15,000 |
| 2 | 💎 High Cost | ₹16,000 – ₹25,000 |
| 3 | 🏆 Very High Cost | ₹26,000+ |

---

## 🚀 Features
- Machine learning model trained using **Random Forest Classifier**
- ~**89% Accuracy** on validation data
- **Streamlit Web App** for interactive predictions
- **Feature Importance Visualization** to explain model decisions

---

## 🧠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-Learn
- Matplotlib
- Streamlit

---

## 📊 Feature Importance Example
The app displays a horizontal bar chart showing which features influence the price the most (e.g., RAM, Battery Power, Pixel Density).

---

## 🏗️ How to Run
```bash
pip install -r requirements.txt
streamlit run Mobile_Price_Prediction_App.py
