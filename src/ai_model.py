# src/ai_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler

# ML Imports
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# DL Imports
try:
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_DL = True
except ImportError:
    HAS_DL = False

def create_lstm_model(input_shape):
    """Builds a simple LSTM suitable for small crypto datasets"""
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2)) 
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

class CryptoModel:
    def __init__(self, model_type='ensemble', lookback=10):
        self.model_type = model_type
        self.lookback = lookback  # Store lookback centrally
        self.features = []
        self.scaler = MinMaxScaler()
        
        # We delay model creation for LSTM because it needs input_shape from data
        if model_type != 'lstm':
            self.model = self._get_model_instance(model_type)
        else:
            self.model = None

    def _get_model_instance(self, model_type):
        """Factory method to create the right model."""
        
        # 1. Define Base Classifiers
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, scale_pos_weight=5, eval_metric='logloss') if XGBClassifier else None
        lgbm = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, is_unbalance=True, verbose=-1) if LGBMClassifier else None

        # 2. Return the requested one
        if model_type == 'rf':
            return rf
        elif model_type == 'xgb':
            if not xgb: raise ImportError("Pip install xgboost")
            return xgb
        elif model_type == 'lgbm':
            if not lgbm: raise ImportError("Pip install lightgbm")
            return lgbm
        
        # 3. Handle Ensemble
        elif model_type == 'ensemble':
            estimators = [('rf', rf)]
            if xgb: estimators.append(('xgb', xgb))
            if lgbm: estimators.append(('lgbm', lgbm))
            
            print(f"--- 🤝 Creating Ensemble with: {[name for name, _ in estimators]} ---")
            return VotingClassifier(estimators=estimators, voting='soft')
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def prepare_dl_data(self, df, is_training=True):
        """
        Prepares 3D data for LSTM: [Samples, TimeSteps, Features]
        """
        # Exclude non-feature columns
        cols = [c for c in df.columns if c not in ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # --- SCALING LOGIC ---
        if is_training:
            self.features = cols
            # Fit scaler on training data
            scaled_data = self.scaler.fit_transform(df[cols])
        else:
            # Use existing scaler for Test/Prediction
            if not self.features: raise Exception("Model not trained, features unknown")
            scaled_data = self.scaler.transform(df[self.features])
            
        data = scaled_data
        
        # Handle Target (Safe extraction)
        if 'target' in df.columns:
            target = df['target'].values
        else:
            # If no target exists (Live mode), create a dummy array of Nones
            target = [None] * len(df)

        X, y = [], []
        
        # --- CREATE SEQUENCES ---
        # We now generate sequences for EVERYTHING passed to this function.
        # We rely on the calling function to pick the right one.
        if len(data) < self.lookback:
             raise ValueError(f"Not enough data! Need {self.lookback} rows, got {len(data)}")

        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i]) 
            y.append(target[i])          
            
        return np.array(X), np.array(y)

    def train(self, df):
        cols_to_drop = ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'SMA_50']

        if self.model_type == 'lstm':
            if not HAS_DL: raise ImportError("Pip install tensorflow")
            print("--- 🧠 Training Deep Learning (LSTM) ---")
            
            # 1. Split Data
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            # 2. Prepare Data
            X_train, y_train = self.prepare_dl_data(train_df, is_training=True)
            X_test, y_test = self.prepare_dl_data(test_df, is_training=False)
            
            # --- NEW: CALCULATE CLASS WEIGHTS ---
            from sklearn.utils import class_weight
            # We look at y_train to see how rare the "1s" are
            weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            # Convert to dictionary {0: 1.0, 1: 5.0}
            class_weights_dict = dict(enumerate(weights))
            print(f"⚖️ Class Weights: {class_weights_dict}")
            # ------------------------------------

            # Build
            self.model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Train (Add class_weight parameter)
            self.model.fit(
                X_train, y_train, 
                epochs=20, 
                batch_size=32, 
                validation_data=(X_test, y_test), 
                verbose=1,
                class_weight=class_weights_dict # <--- THE FIX
            )
            
            # Evaluate
            loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"\nLSTM Accuracy: {acc:.2%}")

        

        else:
            # STANDARD ML TRAINING
            X = df.drop(columns=cols_to_drop, errors='ignore')
            y = df['target']
            self.features = X.columns.tolist()
            
            print(f"Training {self.model_type.upper()} on {len(self.features)} features...")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            self.model.fit(X_train, y_train)
            
            predictions = self.model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            print(f"\n{self.model_type.upper()} Accuracy: {acc:.2%}")
            print(classification_report(y_test, predictions))

    def predict_signal(self, current_data, threshold=0.7):
        if not self.features: raise Exception("Model not trained!")

        if self.model_type == 'lstm':
            X_seq, _ = self.prepare_dl_data(current_data, is_training=False)
            if len(X_seq) == 0: return 0, 0.0 # Return tuple

            last_sequence = X_seq[-1:] 
            prob = self.model.predict(last_sequence, verbose=0)[0][0]
            
            # Return Decision (1/0) AND Probability (0.0 to 1.0)
            return (1 if prob >= threshold else 0), prob

        else:
            data = current_data[self.features].tail(1)
            probs = self.model.predict_proba(data)[0]
            # probs[1] is the Buy Probability
            return (1 if probs[1] >= threshold else 0), probs[1]

    def save_model(self, filename=None):
        # 1. Setup Data Folder
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # 2. Set Default Name if None
        if filename is None: 
            filename = f"model_{self.model_type}"
            
        # 3. Force path to be inside data/ (if user didn't provide it)
        if 'data' not in filename:
            filepath = os.path.join('data', filename)
        else:
            filepath = filename

        # 4. Prepare Metadata
        meta_data = {
            'features': self.features,
            'type': self.model_type,
            'scaler': self.scaler,
            'lookback': self.lookback
        }
        
        # 5. Save (Different logic for LSTM vs ML)
        if self.model_type == 'lstm':
            # Clean extensions just in case
            base_path = filepath.replace('.keras', '').replace('.pkl', '')
            
            model_file = base_path + ".keras" 
            meta_file = base_path + "_meta.pkl"
            
            self.model.save(model_file)
            joblib.dump(meta_data, meta_file)
            print(f"💾 Saved LSTM to {model_file} and {meta_file}")
            
        else:
            # Standard ML Models (XGB, RF, Ensemble)
            if not filepath.endswith('.pkl'): 
                filepath += '.pkl'
                
            meta_data['model'] = self.model
            joblib.dump(meta_data, filepath)
            print(f"💾 Saved {self.model_type} model to {filepath}")

    def load_model(self, filename_base):
        # 1. Force path to be inside data/
        if 'data' not in filename_base:
            filepath = os.path.join('data', filename_base)
        else:
            filepath = filename_base

        # 2. Check for LSTM indicators
        if 'lstm' in filepath.lower() or 'keras' in filepath.lower():
            # Clean up extensions to get the "Base" path
            clean_base = filepath.replace('.keras', '').replace('.h5', '').replace('_meta.pkl', '').replace('.pkl', '')
            
            # Try loading .keras
            model_path = clean_base + ".keras"
            meta_path = clean_base + "_meta.pkl"
            
            if not os.path.exists(model_path):
                 # Fallback for older .h5 models
                model_path = clean_base + ".h5"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ LSTM Model not found at {model_path}")

            self.model = keras_load_model(model_path)
            meta = joblib.load(meta_path)
            
            self.features = meta['features']
            self.model_type = meta['type']
            self.scaler = meta['scaler']
            self.lookback = meta.get('lookback', 10)
            print(f"✅ Loaded LSTM from {model_path}")
            
        else:
            # Standard ML Models
            if not filepath.endswith('.pkl'):
                filepath += '.pkl'
                
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"❌ Model file not found at {filepath}")

            data = joblib.load(filepath)
            self.model = data['model']
            self.features = data['features']
            self.model_type = data.get('type', 'unknown')
            self.scaler = data.get('scaler') # Important to load scaler!
            print(f"✅ Loaded {self.model_type} from {filepath}")

    def get_feature_importance(self):
        if self.model_type in ['lgbm', 'rf', 'xgb']:
            if hasattr(self.model, 'feature_importances_'):
                imp = self.model.feature_importances_
            elif hasattr(self.model, 'estimators_'): 
                return pd.DataFrame() 
            else:
                return pd.DataFrame()

            feats = pd.DataFrame({'Feature': self.features, 'Importance': imp})
            return feats.sort_values(by='Importance', ascending=False)
        return pd.DataFrame()