import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset = [
    ['Roti', 'Selai', 'Mentega'],
    ['Roti', 'Mentega'],
    ['Roti', 'Susu', 'Mentega'],
    ['Cokelat', 'Roti', 'Susu', 'Mentega'],
    ['Cokelat', 'Susu']
]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

print("Frequent Itemsets:")
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print("\nAssociation Rules:")
print(rules)
