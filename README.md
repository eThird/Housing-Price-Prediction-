
```markdown
# Housing Price Prediction using Machine Learning

A regression-based machine learning project to predict housing prices using various features like number of rooms, location, crime rate, and more. This end-to-end project includes data preprocessing, model training, evaluation, and prediction.

##  Project Goal

To build a predictive model using supervised learning that can accurately estimate housing prices based on input features from the Boston Housing dataset.

---

## Project Structure

```

Housing-Price-Prediction-/
├── dataset/
│   └── Boston.csv
├── visuals/
│   └── correlation\_matrix.png
├── housing\_model.pkl
├── Housing\_Price\_Predictor.ipynb
├── requirements.txt
└── README.md

````

---

## Features Used

The dataset includes the following key features:

- `CRIM` — Crime rate per capita
- `ZN` — Proportion of residential land zoned
- `INDUS` — Proportion of non-retail business acres
- `CHAS` — Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- `NOX` — Nitric oxide concentration
- `RM` — Average number of rooms
- `AGE` — Proportion of owner-occupied units built before 1940
- `DIS` — Weighted distances to employment centers
- `RAD`, `TAX`, `PTRATIO` — Indexes related to infrastructure and taxation
- `LSTAT` — % of lower status population
- `MEDV` — Median value of owner-occupied homes (Target variable)

---

## 🧠 ML Workflow

1. **Data Cleaning** – Null checks, duplicate removal
2. **Exploratory Data Analysis** – Feature distributions, correlation heatmaps
3. **Feature Selection** – Based on correlation and domain knowledge
4. **Model Selection** – Linear Regression, Ridge, Lasso
5. **Model Evaluation** – R² Score, RMSE
6. **Model Serialization** – Model saved as `housing_model.pkl` using `joblib`

---

## 🛠️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/eThird/Housing-Price-Prediction-.git
cd Housing-Price-Prediction-
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook Housing_Price_Predictor.ipynb
```

---

## 🔍 Example Prediction (Coming Soon)

```python
# Load model and make prediction
import joblib
model = joblib.load('housing_model.pkl')
prediction = model.predict([[0.02, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 4.98, 24.0]])
print(f"Predicted Price: ${prediction[0]*1000:.2f}")
```

---

📈 Sample Visualization
<table> <tr> <td><img src="https://github.com/user-attachments/assets/8fede4e0-79c7-4c13-9eae-dc9b0f27bcbd" width="300"/></td> <td><img src="https://github.com/user-attachments/assets/840bf261-cc1a-4805-887a-c0e48ef0517b" width="300"/></td> <td><img src="https://github.com/user-attachments/assets/820d974e-aeae-4c99-ac17-e126d304dc3a" width="300"/></td> <td><img src="https://github.com/user-attachments/assets/cf34e009-6925-43ff-a906-2599ace4456c" width="300"/></td> </tr> </table>

---

## ✅ Tech Stack

* **Language**: Python
* **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib
* **Tools**: Jupyter Notebook

---

## 🧠 Learning Outcomes

* Feature engineering and preprocessing techniques
* Regression model tuning and comparison
* Model serialization and reusability
* Visual analysis of high-dimensional data

---

## 🤝 Contribution

This is a solo project by [Pranshu Singh](https://github.com/eThird).
Contributions and suggestions are welcome. Feel free to open an issue or pull request.

---



## 🔗 Connect with Me

* LinkedIn: [linkedin.com/in/pranshu24](https://www.linkedin.com/in/pranshu24)
* GitHub: [github.com/eThird](https://github.com/eThird)

```


