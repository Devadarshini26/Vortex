
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/MyDrive/Social_Network_Ads.csv')

bins = [0, 30, 40, 50, 60, float('inf')]
labels = ['0-30', '31-40', '41-50', '51-60', '61+']
data['Age_Category'] = pd.cut(data['Age'], bins=bins, labels=labels)

salary_bins = [0, 50000, 100000, 150000, float('inf')]
salary_labels = ['0-50K', '50K-100K', '100K-150K', '150K+']
data['Salary_Category'] = pd.cut(data['EstimatedSalary'], bins=salary_bins, labels=salary_labels)

data.drop(['Age', 'EstimatedSalary'], axis=1, inplace=True)

data_onehot = pd.get_dummies(data)

frequent_itemsets = apriori(data_onehot, min_support=0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Apriori Association Rules')
plt.show()
