import pandas as pd
import joblib
import sys
import os

# Add project root to path so we can import the class
sys.path.append(os.getcwd())

def inspect_ensemble_votes():
    # 1. Load the "Mega-Brain" (Ensemble)
    model_path = 'data/model_ensemble.pkl' # Ensure this matches your save name
    if not os.path.exists(model_path):
        print("❌ Ensemble model not found. Train it first!")
        return

    print(f"--- 🕵️ Inspecting {model_path} ---")
    data = joblib.load(model_path)
    ensemble = data['model'] # This is the VotingClassifier
    features = data['features']
    
    # 2. Check if it is actually a VotingClassifier
    if not hasattr(ensemble, 'estimators_'):
        print("This is not an Ensemble model. It's a single model.")
        return

    # 3. Load some test data (Recent data)
    df = pd.read_excel('data/btc_daily_2010_2025_processed.xlsx')
    # Let's look at the last 5 days
    recent_data = df.tail(5).copy()
    X_test = recent_data[features]
    
    # 4. Get Individual Votes
    # ensemble.estimators_ is a list of the 3 trained models [RF, XGB, LGBM]
    print("\n--- 🗳️ VOTING BREAKDOWN (Last 5 Days) ---")
    
    for i in range(len(X_test)):
        row = X_test.iloc[[i]]
        date = recent_data.iloc[i]['timestamp']
        
        print(f"\n📅 Date: {date}")
        
        # Get the vote from each expert
        votes = []
        for name, clf in ensemble.named_estimators_.items():
            # clf.predict returns [0] or [1]
            vote = clf.predict(row)[0]
            prob = clf.predict_proba(row)[0][1] # Confidence to Buy
            
            vote_str = "🟢 BUY " if vote == 1 else "🔴 WAIT"
            print(f"   {name.upper()}: {vote_str} (Conf: {prob:.1%})")
            votes.append(vote)
            
        # Get the Final Ensemble Decision
        final_vote = ensemble.predict(row)[0]
        final_str = "🟢 BUY " if final_vote == 1 else "🔴 WAIT"
        print(f"   👉 FINAL DECISION: {final_str}")

if __name__ == "__main__":
    inspect_ensemble_votes()