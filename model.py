# model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_splendour_core_model():
    # 7 Premium Nigerian Market Leaders traded on the exchange
    companies = ['MTNN', 'ZENITHBANK', 'GTCO', 'DANGCEM', 'BUACEMENT', 'NESTLE', 'ACCESSCORP']
    np.random.seed(42)
    days = 250
    data_list = []
    
    for company in companies:
        base_price = {
            'MTNN': 840.00, 
            'ZENITHBANK': 128.50, 
            'GTCO': 139.00, 
            'DANGCEM': 1220.00,
            'BUACEMENT': 112.00,
            'NESTLE': 910.00,
            'ACCESSCORP': 64.50
        }[company]
        
        for day in range(days):
            price_fluctuation = np.random.normal(0, base_price * 0.015)
            current_price = max(5.00, base_price + price_fluctuation + (day * 0.08))
            
            rsi = np.random.uniform(28, 82)
            moving_avg_50 = current_price * np.random.uniform(0.96, 1.04)
            volume = int(np.random.uniform(400000, 12000000))
            
            # Binary strategy target rules mapping normal vs consolidation states
            target = 1 if (current_price > moving_avg_50 and rsi < 65) else 0
            
            data_list.append({
                'company': company,
                'price': round(current_price, 2),
                'rsi': round(rsi, 2),
                'ma50': round(moving_avg_50, 2),
                'volume': volume,
                'target': target
            })
            
    df = pd.DataFrame(data_list)
    X = df[['price', 'rsi', 'ma50', 'volume']]
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    with open('splendour_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Splendour Analyzer Engine successfully trained and serialized locally.")

if __name__ == '__main__':
    train_splendour_core_model()