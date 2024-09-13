import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('fraud_data.csv')

# Clean 'is_fraud' column to keep only 0 and 1
df['is_fraud'] = df['is_fraud'].str.extract(r'(\d)').astype(int)

# Group transactions by category to calculate total and fraud transactions. Used this to help write the code: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.aggregate.html
category_data = df.groupby('category').agg(total_transactions=('is_fraud', 'count'), fraud_transactions=('is_fraud', 'sum')).reset_index()

# Calculate fraud rate by category
category_data['fraud_rate'] = category_data['fraud_transactions'] / category_data['total_transactions']

# Calculate age. Used this to help write the code: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
df['dob'] = pd.to_datetime(df['dob'], dayfirst=True)
df['age'] = pd.to_datetime('today').year - df['dob'].dt.year

# Group by age. Used this to help write the code: https://meet1291.medium.com/how-to-use-pandas-pd-cut-with-age-example-2773676e95e
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 60, 100], labels=['<25', '25-35', '35-45', '45-60', '60+'])
age_group_fraud = df[df['is_fraud'] == 1].groupby('age_group').size()

plt.figure(figsize=(14, 14))

# Fraud rate by transaction category
plt.subplot(2, 2, 1)
sns.barplot(x='category', y='fraud_rate', data=category_data, palette="Blues_d")
plt.title('Fraud Rate by Transaction Category')
plt.ylabel('Fraud Rate')
plt.xlabel('Category')
plt.xticks(rotation=90)

# Age groups targeted by credit card fraud
plt.subplot(2, 2, 2)
sns.barplot(x=age_group_fraud.index, y=age_group_fraud.values, palette="Blues_d")
plt.title('Age Group Targeted by Credit Card Fraud')
plt.ylabel('Number of Fraudulent Transactions')
plt.xlabel('Age Group')

# Correlation between transaction amount and fraud
plt.subplot(2, 1, 2)
sns.scatterplot(x='amt', y='is_fraud', data=df, alpha=0.5, palette='Blues_d')
plt.title('Correlation between Transaction Amount and Fraud')
plt.xlabel('Transaction Amount')
plt.ylabel('Is Fraudulent')

plt.tight_layout()
plt.show()
