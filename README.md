# 📘 Experiment – 14

## Data Normalization and Data Type Conversion using Python

**Name:** Dev Anand
**PRN:** 25070123039
**Batch:** A2 ENTC

---

# 📌 Introduction

Data preprocessing is a crucial step in data analysis and machine learning.

This experiment focuses on:

* **Data Normalization** (scaling values)
* **Data Type Conversion** (encoding categorical data)

These techniques help improve model performance and data consistency.

---

# 🔹 PART A – Data Normalization

## 📊 Dataset: Product Data

```python
import pandas as pd
import numpy as np
```

Dataset contains:

* Product
* Price
* Units Sold
* Discount

---

# 🧪 Normalization Techniques

## 1️⃣ Min-Max Normalization

Formula:

```
(x - min) / (max - min)
```

### Price Normalization

```python
df['Price_MinMax'] = (df['Price'] - df['Price'].min()) / (df['Price'].max() - df['Price'].min())
```

### Units Sold Normalization

```python
df['Units_sold_MinMax'] = (df['Units_sold'] - df['Units_sold'].min()) / (df['Units_sold'].max() - df['Units_sold'].min())
```

### Discount Normalization

```python
df['Discount_MinMax'] = (df['Discount'] - df['Discount'].min()) / (df['Discount'].max() - df['Discount'].min())
```

📘 Scales values between 0 and 1.

---

## 2️⃣ Z-Score Normalization

Formula:

```
(x - mean) / standard deviation
```

```python
df['Units_Zscore'] = (df['Units_sold'] - df['Units_sold'].mean()) / df['Units_sold'].std()
```

📘 Centers data around mean = 0 and std = 1.

---

## 3️⃣ Decimal Scaling

```python
df['Price_Decimal'] = df['Price'] / 100000
```

```python
df['Discount_Decimal'] = df['Discount'] / 100
```

📘 Moves decimal point to scale values.

---

## 🔹 Normalize Multiple Columns

```python
cols = ['Price','Units_sold','Discount']

df_norm = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
```

📘 Applies normalization to multiple features.

---

# 🔹 PART B – Data Type Conversion

## 📊 Dataset: Order Data

Contains:

* Order_ID
* Gender
* Product Method
* Category
* City
* Order Value

---

# 🧪 Encoding Techniques

## 1️⃣ Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender_Label'] = le.fit_transform(df['Customer_Gender'])
```

📘 Converts categories into numeric labels.

---

## 2️⃣ One-Hot Encoding (OHE)

```python
df_encoded = pd.get_dummies(df, columns=['Product_Method'])
```

📘 Creates binary columns for each category.

---

## 3️⃣ Encoding Multiple Columns

```python
df_multi = pd.get_dummies(df, columns=['Product_Category','City'])
```

---

## 4️⃣ Dummy Encoding (Drop First)

```python
df_dummy = pd.get_dummies(df, columns=['Product_Method'], drop_first=True)
```

📘 Avoids multicollinearity.

---

# 🔹 New Dataset Operations

## 📂 Amazon Dataset

### Min-Max Normalization

```python
cols = ['Price','Units_Sold','Reviews','Rating']

df_norm = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
```

---

### Z-Score Normalization

```python
df_norm = (df[cols] - df[cols].mean()) / df[cols].std()
```

---

### Decimal Scaling

```python
df['Price_Decimal'] = df['Price'] / 100
```

---

## 📂 Student Dataset

### Label Encoding

```python
le = LabelEncoder()
df['Placement_Status'] = le.fit_transform(df['Gender'])
```

---

### One-Hot Encoding

```python
df_encoded = pd.get_dummies(df, columns=['Department'])
```

---

# 🎯 Learning Outcomes

After this experiment, the student can:

* Apply normalization techniques
* Scale data effectively
* Convert categorical data into numeric form
* Prepare datasets for machine learning

---

# 📚 Applications

* Machine learning preprocessing
* Data analysis
* Feature engineering
* Business analytics

---

# ✅ Conclusion

Data normalization and type conversion are essential for:

* Improving model accuracy
* Ensuring uniform data scale
* Converting categorical data into usable formats

Using Pandas and Scikit-learn, these tasks can be efficiently performed.

---

## ✍ Author

Dev Anand
B.Tech ENTC – Symbiosis Institute of Technology, Pune

---


