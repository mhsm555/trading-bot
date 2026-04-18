import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

# Suppress warnings
warnings.filterwarnings('ignore')

# --- OPTIONAL IMPORTS ---
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    HAS_DL = True
except ImportError:
    HAS_DL = False

class CryptoModel:
    def __init__(self, model_type='ensemble', lookback=10):
        self.model_type = model_type
        self.lookback = lookback
        self.features = []
        self.scaler = MinMaxScaler()
        self.model = None
        
        # Initialize non-DL models immediately
        if model_type != 'lstm':
            self.model = self._get_model_instance(model_type)

    def _get_model_instance(self, model_type):
        """Factory to create the requested ML model."""
        rf = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight='balanced', random_state=42, n_jobs=-1)
        
        xgb = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05, 
            scale_pos_weight=3, eval_metric='logloss', n_jobs=-1
        ) if XGBClassifier else None
        
        lgbm = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05, 
            class_weight='balanced', verbosity=-1, n_jobs=-1
        ) if LGBMClassifier else None

        if model_type == 'rf': return rf
        elif model_type == 'xgb':
            if not xgb: raise ImportError("❌ XGBoost not installed. Run: pip install xgboost")
            return xgb
        elif model_type == 'lgbm':
            if not lgbm: raise ImportError("❌ LightGBM not installed. Run: pip install lightgbm")
            return lgbm
        elif model_type == 'ensemble':
            estimators = [('rf', rf)]
            if xgb: estimators.append(('xgb', xgb))
            if lgbm: estimators.append(('lgbm', lgbm))
            print(f"--- 🤝 Ensemble Strategy: {[name for name, _ in estimators]} ---")
            return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _build_lstm(self, input_shape):
        if not HAS_DL: raise ImportError("❌ TensorFlow not installed.")
        
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def prepare_dl_data(self, df, is_training=True):
        """
        Creates (Samples, Lookback, Features) for LSTM.
        Handles scaling internally to prevent leakage.
        """
        # 1. Filter Columns
        exclude = {'target', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'time'}
        # If we already have trained features, strictly use those. Otherwise, infer them.
        if self.features and not is_training:
            feature_cols = self.features
        else:
            feature_cols = [c for c in df.columns if c not in exclude]

        # 2. Vectorized Sequence Generation
        # Note: We scale AFTER splitting in the train loop, so here we just process raw data
        # UNLESS this is live prediction, in which case we scale here.
        
        if is_training:
            # In training, we return RAW data and let the train() method handle scaling/splitting
            # This is a change from your original code to fix the leakage.
            return df[feature_cols].values, df['target'].values if 'target' in df.columns else None
        
        else:
            # LIVE PREDICTION: Data comes in, we scale it immediately
            data_values = df[feature_cols].values
            scaled_data = self.scaler.transform(data_values)
            
            # Create Window
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                X = sliding_window_view(scaled_data, window_shape=self.lookback, axis=0)
                X = np.swapaxes(X, 1, 2)
            except ImportError:
                X = []
                for i in range(self.lookback, len(scaled_data) + 1):
                    X.append(scaled_data[i-self.lookback:i])
                X = np.array(X)
                
            return X, None

    def _create_windows(self, data, lookback):
        """Helper to create windows from numpy array"""
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            X = sliding_window_view(data, window_shape=lookback, axis=0)
            return np.swapaxes(X, 1, 2)
        except ImportError:
            X = []
            for i in range(lookback, len(data) + 1):
                X.append(data[i-lookback:i])
            return np.array(X)

    def train(self, df):
        """Universal Train Method with NO DATA LEAKAGE"""
        print(f"🧠 Training {self.model_type.upper()} Model...")

        # 1. Define Features
        exclude = {'target', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'time'}
        self.features = [c for c in df.columns if c not in exclude]
        print(f"📊 Features ({len(self.features)}): {self.features}")

        # 2. Time-Series Split (80/20) - BEFORE SCALING
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # 3. Fit Scaler ONLY on Training Data
        X_train_raw = train_df[self.features].values
        X_test_raw = test_df[self.features].values
        
        self.scaler.fit(X_train_raw)
        
        X_train_scaled = self.scaler.transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)
        
        y_train = train_df['target'].values
        y_test = test_df['target'].values

        # --- PATH A: Deep Learning (LSTM) ---
        if self.model_type == 'lstm':
            # Create Windows
            X_train_lstm = self._create_windows(X_train_scaled, self.lookback)
            X_test_lstm = self._create_windows(X_test_scaled, self.lookback)
            
            # Align Targets (Trim start to match windows)
            # If window size is 10, we lose first 9 labels
            y_train_lstm = y_train[self.lookback-1:]
            y_test_lstm = y_test[self.lookback-1:]
            
            # Safety Trim
            min_len_train = min(len(X_train_lstm), len(y_train_lstm))
            X_train_lstm, y_train_lstm = X_train_lstm[:min_len_train], y_train_lstm[:min_len_train]
            
            min_len_test = min(len(X_test_lstm), len(y_test_lstm))
            X_test_lstm, y_test_lstm = X_test_lstm[:min_len_test], y_test_lstm[:min_len_test]

            # Weights
            try:
                weights = class_weight.compute_class_weight(
                    class_weight='balanced', classes=np.unique(y_train_lstm), y=y_train_lstm
                )
                class_weights = dict(enumerate(weights))
            except:
                class_weights = {0: 1.0, 1: 1.0}

            # Build & Fit
            self.model = self._build_lstm(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
            self.model.fit(
                X_train_lstm, y_train_lstm,
                epochs=20, batch_size=64,
                validation_data=(X_test_lstm, y_test_lstm),
                class_weight=class_weights, verbose=1
            )

        # --- PATH B: Machine Learning ---
        else:
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            preds = self.model.predict(X_test_scaled)
            acc = accuracy_score(y_test, preds)
            print(f"🏆 {self.model_type.upper()} Test Accuracy: {acc:.2%}")

    def predict_signal(self, current_data, threshold=0.7):
        if not self.features: return 0, 0.0

        try:
            # 1. Prepare Data
            df_subset = current_data[self.features].copy()
            for col in df_subset.columns:
                df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
            df_subset = df_subset.ffill().fillna(0).replace([np.inf, -np.inf], 0)

            # 2. Check Data Length (LSTM only)
            if self.model_type == 'lstm' and len(df_subset) < self.lookback:
                return 0, 0.0

            # 3. Scale & Predict
            if self.model_type == 'lstm':
                recent_df = df_subset.tail(self.lookback)
                scaled = self.scaler.transform(recent_df)
                
                if np.isnan(scaled).any(): scaled = np.nan_to_num(scaled)
                
                # Reshape: (1, Lookback, Features)
                X_live = scaled.reshape(1, self.lookback, len(self.features)).astype(np.float32)
                prob = self.model.predict(X_live, verbose=0)[0][0]
                return (1 if prob >= threshold else 0), float(prob)
            else:
                last_row = df_subset.tail(1)
                scaled = self.scaler.transform(last_row)
                probs = self.model.predict_proba(scaled)[0]
                return (1 if probs[1] >= threshold else 0), float(probs[1])

        except Exception as e:
            print(f"❌ Predict Error: {e}")
            return 0, 0.0

    def save_model(self, filename=None):
        if not os.path.exists('data'): os.makedirs('data')
        if filename is None: filename = f"model_{self.model_type}"
        filepath = os.path.join('data', filename) if 'data' not in filename else filename

        meta = {
            'features': self.features,
            'type': self.model_type,
            'scaler': self.scaler,
            'lookback': self.lookback
        }

        if self.model_type == 'lstm':
            base = filepath.replace('.keras', '').replace('.pkl', '')
            self.model.save(f"{base}.keras")
            joblib.dump(meta, f"{base}_meta.pkl")
            print(f"💾 Saved LSTM: {base}.keras")
        else:
            if not filepath.endswith('.pkl'): filepath += '.pkl'
            meta['model'] = self.model
            joblib.dump(meta, filepath)
            print(f"💾 Saved ML Model: {filepath}")

    def load_model(self, filename_base):
        filepath = os.path.join('data', filename_base) if 'data' not in filename_base else filename_base
        clean_base = filepath.replace('.keras', '').replace('.pkl', '').replace('_meta', '')
        
        # Check for LSTM
        if os.path.exists(f"{clean_base}.keras") or os.path.exists(f"{clean_base}.h5"):
            meta_path = f"{clean_base}_meta.pkl"
            model_path = f"{clean_base}.keras" if os.path.exists(f"{clean_base}.keras") else f"{clean_base}.h5"
            self.model = keras_load_model(model_path)
            meta = joblib.load(meta_path)
            print(f"✅ Loaded LSTM from {model_path}")
        else:
            if not filepath.endswith('.pkl'): filepath += '.pkl'
            if not os.path.exists(filepath): raise FileNotFoundError(f"❌ Model not found: {filepath}")
            data = joblib.load(filepath)
            self.model = data['model']
            meta = data
            print(f"✅ Loaded ML Model from {filepath}")

        self.features = meta['features']
        self.model_type = meta['type']
        self.scaler = meta['scaler']
        self.lookback = meta.get('lookback', 10)