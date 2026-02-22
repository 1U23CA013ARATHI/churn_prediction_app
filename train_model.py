import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Data Load
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Basic Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 3. Feature Selection (4 main features)
X = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']]
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# 4. Encoding Contract (String to Number)
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
X['Contract'] = X['Contract'].map(contract_map)

# 5. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Saving the Brain (Model)
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as churn_model.pkl!")