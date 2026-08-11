import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def detect_feature_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, threshold: float = 0.05) -> dict:
    """Compare baseline reference feature distributions with new incoming inference data."""
    drift_report = {}
    features = ['MACD', 'RSI_14', 'Volatility_14', 'Log_Return']

    for col in features:
        if col in reference_data.columns and col in current_data.columns:
            # Kolmogorov-Smirnov 2-sample test
            stat, p_value = ks_2samp(reference_data[col], current_data[col])
            is_drifted = p_value < threshold
            drift_report[col] = {
                "p_value": float(p_value),
                "ks_stat": float(stat),
                "drift_detected": bool(is_drifted)
            }
            
    return drift_report

if __name__ == "__main__":
    # Generate synthetic reference and drifted sample
    np.random.seed(42)
    ref = pd.DataFrame({"RSI_14": np.random.normal(50, 10, 1000), "MACD": np.random.normal(0, 1, 1000)})
    curr = pd.DataFrame({"RSI_14": np.random.normal(65, 15, 200), "MACD": np.random.normal(0, 1, 200)}) # Shifted mean
    
    report = detect_feature_drift(ref, curr)
    print("Drift Monitoring Report:", report)