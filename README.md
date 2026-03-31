## 📊 Dataset Creation

```python
import pandas as pd
import numpy as np

data={
    'Product':['Laptop','Mobile','Tablet','Headphones','Smartwatch','Camera'],
    'Price':[55000,20000,30000,2000,8000,45000],
    'Units_sold':[25,60,35,80,50,15],
    'Discount':[10,15,12,20,18,8]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)
🔸 1. Min-Max Normalization

Formula:

(x - min) / (max - min)
df['Price_MinMax']=(df['Price']-df['Price'].min())/(df['Price'].max()-df['Price'].min())

df['Units_sold_MinMax']=(df['Units_sold']-df['Units_sold'].min())/(df['Units_sold'].max()-df['Units_sold'].min())

df['Discount_MinMax']=(df['Discount']-df['Discount'].min())/(df['Discount'].max()-df['Discount'].min())
🔸 2. Z-Score Normalization

Formula:

(x - mean) / std
df['Units_Zscore']= (df['Units_sold']-df['Units_sold'].mean()) / df['Units_sold'].std()
🔸 3. Decimal Scaling
df['Price_Decimal'] = df['Price']/100000
df['Discount_Decimal'] = df['Discount']/100
🔸 Normalize Multiple Columns
cols = ['Price','Units_sold','Discount']

df_norm =(df[cols] - df[cols].min())/(df[cols].max() - df[cols].min())

print(df_norm)
🔹 Part B: Data Type Conversion
📖 What is Data Type Conversion?

Categorical data must be converted into numerical format for analysis and machine learning.

📊 Dataset
data = {
    'Order_ID':[101,102,103,104,105,106],
    'Customer_Gender':['Male','Female','Female','Male','Female','Male'],
    'Product_Method':['UPI','Credit Card','Debit Card','COD','UPI','Debit Card'],
    'Product_Category':['Electronics','Clothing','Home','Electronics','Beauty','Electronics'],
    'City':['Pune','Mumbai','Delhi','Bangalore','Pune','Hyderabad'],
    'Order_Value':[4500,2300,3200,5200,1800,4100]
}

df = pd.DataFrame(data)

print(df)
🔸 1. Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender_Label'] = le.fit_transform(df['Customer_Gender'])
🔸 2. One-Hot Encoding
df_encoded = pd.get_dummies(df,columns=['Product_Method'])
🔸 3. Encoding Multiple Columns
df_multi = pd.get_dummies(df,columns=['Product_Category','City'])
🔸 4. Dummy Encoding (Drop First)
df_dummy = pd.get_dummies(df,columns=['Product_Method'],drop_first=True)
🔹 Real Dataset Operations
📥 Load Dataset
df = pd.read_csv("/content/amazon_products_dataset_Expt-14.csv")
🔸 Min-Max Normalization
cols = ['Price','Units_Sold','Reviews','Rating']

df_norm =(df[cols] - df[cols].min())/(df[cols].max() - df[cols].min())
🔸 Z-Score Normalization
df_norm= (df[cols]-df[cols].mean()) / df[cols].std()
🔸 Decimal Scaling
df['Price_Decimal'] = df['Price']/100
🔹 Data Conversion on Student Dataset
df = pd.read_csv("/content/Student-Dataset.csv")
🔸 Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Placement_Status'] = le.fit_transform(df['Gender'])
🔸 One-Hot Encoding
df_encoded = pd.get_dummies(df,columns=['Department'])
🎯 Learning Outcomes
Understand normalization techniques
Apply Min-Max, Z-score, and Decimal Scaling
Convert categorical data into numerical form
Perform encoding techniques
Work with real-world datasets
📚 Applications
Machine Learning
Data Science
Business Analytics
E-commerce systems
Financial analysis
Healthcare data
✅ Conclusion

Data preprocessing is a crucial step in data analysis.

Using Pandas, we can normalize numerical data, convert categorical data, and prepare datasets for machine learning efficiently.

✍ Author

Dev Anand
B.Tech ENTC
Symbiosis Institute of Technology, Pune
