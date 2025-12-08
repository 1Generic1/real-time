# advanced_ml_predictor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")


class AdvancedMLPredictor:
    def __init__(self):
        # Random Forest model
        self.rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
        # LSTM model
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.sequence_length = 30  # Number of time steps for LSTM
        self.is_trained = False

    # --------------------------
    # DATA PREPARATION FUNCTIONS
    # --------------------------
    def prepare_lstm_data(self, price_data: pd.DataFrame):
        """Prepare sequences for LSTM"""
        features = ['open', 'high', 'low', 'close', 'volume']
        data = price_data[features].values
        data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            # Direction: 1 = price up next step, 0 = down
            y.append(1 if data[i, 3] > data[i - 1, 3] else 0)
        return np.array(X), np.array(y)

    def prepare_rf_data(self, price_data: pd.DataFrame):
        """Prepare features for Random Forest"""
        df = price_data.copy()
        df['returns'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['volatility'] = df['returns'].rolling(10).std()
        df = df.dropna()
        X = df[['returns', 'ma5', 'ma10', 'ma20', 'volatility']].values
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # next-step direction
        return X, y

    # --------------------------
    # TRAINING FUNCTIONS
    # --------------------------
    def train_models(self, price_data: pd.DataFrame, onchain_data: dict):
        """
        Train both LSTM and Random Forest models.
        """
        try:
            # --- LSTM ---
            X_lstm, y_lstm = self.prepare_lstm_data(price_data)
            if len(X_lstm) < 50:  # Not enough data
                print("⚠️  Not enough data for LSTM training")
                return False

            self.lstm_model = Sequential()
            self.lstm_model.add(LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=False))
            self.lstm_model.add(Dropout(0.2))
            self.lstm_model.add(Dense(1, activation='sigmoid'))
            self.lstm_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            self.lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=16, verbose=0, callbacks=[early_stop])

            # --- Random Forest ---
            X_rf, y_rf = self.prepare_rf_data(price_data)
            if len(X_rf) < 50:
                print("⚠️  Not enough data for RF training")
                return False

            self.rf_model.fit(X_rf, y_rf)

            self.is_trained = True
            print("✅ ML models trained successfully")
            return True
        except Exception as e:
            print(f"❌ Error training ML models: {e}")
            return False

    # --------------------------
    # PREDICTION FUNCTIONS
    # --------------------------
    def predict(self, price_data: pd.DataFrame, onchain_data: dict):
        """
        Predict next move probabilities using ensemble.
        Returns:
            - predictions: [LSTM_pred, RF_pred, combined_pred]
            - confidence_scores: probabilities
        """
        if not self.is_trained:
            return [0, 0, 0], [0.5, 0.5, 0.5], {}

        # --- LSTM Prediction ---
        X_lstm = price_data[['open', 'high', 'low', 'close', 'volume']].values
        X_lstm = self.scaler.transform(X_lstm)
        if len(X_lstm) < self.sequence_length:
            lstm_pred = 0
            lstm_conf = 0.5
        else:
            seq = X_lstm[-self.sequence_length:].reshape(1, self.sequence_length, 5)
            lstm_conf = float(self.lstm_model.predict(seq, verbose=0)[0][0])
            lstm_pred = 1 if lstm_conf > 0.5 else 0

        # --- Random Forest Prediction ---
        X_rf = self.prepare_rf_data(price_data)[0]
        rf_conf = float(self.rf_model.predict_proba(X_rf[-1].reshape(1, -1))[0][1])
        rf_pred = 1 if rf_conf > 0.5 else 0

        # --- Weighted Ensemble ---
        combined_conf = 0.6 * lstm_conf + 0.4 * rf_conf
        combined_pred = 1 if combined_conf > 0.5 else 0

        predictions = [lstm_pred, rf_pred, combined_pred]
        confidence_scores = [lstm_conf, rf_conf, combined_conf]

        return predictions, confidence_scores, {}

    # --------------------------
    # SIGNAL GENERATION
    # --------------------------
    def generate_ml_signals(self, predictions, confidence_scores, current_price):
        """
        Convert predictions into actionable signals.
        Returns: list of signals, ML boost score
        """
        signals = []
        boost = 0

        if predictions[2] == 1:  # Combined prediction LONG
            signals.append(f"🤖 ML SIGNAL: LONG (Confidence {confidence_scores[2]*100:.1f}%)")
            boost = 2 * confidence_scores[2]  # Add to traditional score
        elif predictions[2] == 0:  # Combined prediction SHORT
            signals.append(f"🤖 ML SIGNAL: SHORT (Confidence {confidence_scores[2]*100:.1f}%)")
            boost = -2 * (1 - confidence_scores[2])

        return signals, boost

    def display_predictions(self, predictions, confidence_scores, signals):
        print("\n🤖 ML PREDICTIONS SUMMARY")
        print(f"   LSTM Prediction: {predictions[0]} | Confidence: {confidence_scores[0]*100:.1f}%")
        print(f"   RF Prediction: {predictions[1]} | Confidence: {confidence_scores[1]*100:.1f}%")
        print(f"   Ensemble Prediction: {predictions[2]} | Confidence: {confidence_scores[2]*100:.1f}%")
        for sig in signals:
            print(f"   {sig}")
