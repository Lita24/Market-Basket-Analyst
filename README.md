# Market-Basket-Analyst
Market basket analysis helps companies identify items frequently purchased together. Market Basket analyst.   This project is to identify purchasing patterns and the relationship between products frequently purchased by customers in cafes. Transaction data obtained from the internet.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
data=pd.read_csv("BreadBasket_DMS.csv")
data.head()
data=data.dropna()
basket = data.groupby(['Transaction', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
basket = basket.applymap(lambda x: 1 if x > 0 else 0)
print(basket.head())
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
print(frequent_itemsets.head())
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
