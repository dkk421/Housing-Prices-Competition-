import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/train.csv')

""" 
print(df.head(20))
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
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(df['SalePrice_log'], bins=50, kde=True)
plt.title('Log SalePrice')

plt.subplot(1,2,2)
sns.boxplot(x=df['SalePrice'])
plt.title('Box plot log')

plt.show()