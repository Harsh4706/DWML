!pip install mlxtend


transactions = [
    ['Harry Potter', 'The Hobbit', '1984'],
    ['Harry Potter', 'The Hobbit', 'The Alchemist'],
    ['Harry Potter', '1984', 'The Alchemist'],
    ['The Hobbit', '1984', 'The Alchemist'],
    ['Harry Potter', 'The Hobbit', '1984', 'The Alchemist'],
    ['Harry Potter', 'The Hobbit'],
    ['Harry Potter', '1984'],
    ['The Hobbit', 'The Alchemist'],
    ['Harry Potter', 'The Alchemist'],
    ['Harry Potter', 'The Hobbit', '1984']
]

print("Number of transactions:", len(transactions))


from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_array, columns=te.columns_)

print(df)


from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(
    df,
    min_support=0.40,
    use_colnames=True
)

print(frequent_itemsets)


from mlxtend.frequent_patterns import association_rules

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.60
)

print(rules[
    ['antecedents', 'consequents', 'support', 'confidence', 'lift']
])