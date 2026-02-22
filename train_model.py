import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Data load
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Advanced Cleaning
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Text categories-ah numbers-ah mathurom
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df['Contract'] = df['Contract'].map(contract_map)

# 3. Features list-ah increase panrom
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']
X = df[features]
y = df['Churn']

# 4. Train and Save
model = RandomForestClassifier()
model.fit(X, y)

with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Upgrade Success! New Brain ready with 4 features.")