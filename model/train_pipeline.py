import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/train.csv')

""" print(df.head(20))
print(df.shape) # (1460, 81)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(df['SalePrice'], bins=50, kde=True)
plt.title('Распределение цен')

plt.subplot(1,2,2)
sns.boxplot(x=df['SalePrice'])
plt.title('Box plot цен')

plt.show() """
 
#------- right-skewed распределение ------
# Использую логарифмическое преобразование, тк оно уменьшает большие числа и увеличивает
#  маленькие, благодаря чему данные становятся более сбалансированными
# ----------------------------------------

df['SalePrice_log'] = np.log1p(df['SalePrice'])


""" plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(df['SalePrice_log'], bins=50, kde=True)
plt.title('Log SalePrice')

plt.subplot(1,2,2)
sns.boxplot(x=df['SalePrice'])
plt.title('Box plot log')

plt.show() """

""" print("Shape:", df.shape)
print("\nDtypes:")
print(df.dtypes.value_counts())

missing = df.isnull().sum().sort_values(ascending=False)
print("\nMissing values:")
print(missing[missing > 0].head(20))

numeric_corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
numeric_corr = numeric_corr.drop(["SalePrice", "SalePrice_log"])
print(numeric_corr.head(10))

sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'])
sns.boxplot(x=df['OverallQual'], y=df['SalePrice']) """

#----- The most important features appear to be OverallQual, GrLivArea, and GarageCars------

# ------------------ BASELINE MODEL ------------------

from sklearn.model_selection import train_test_split

y = df['SalePrice_log']
X = df.drop(['SalePrice', 'SalePrice_log'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))