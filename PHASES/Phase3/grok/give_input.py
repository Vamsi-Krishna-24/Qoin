def project_price(factor_values: dict) -> float:
    """
    Pass raw factor values (exactly as they appear in your CSV).
    Returns projected ICICI price in ₹.
    """
    raw = np.array([factor_values[col] for col in feature_cols]).reshape(1, -1)
    x_scaled = scaler_X.transform(raw)
    y_scaled_pred = model.predict(x_scaled)[0]
    projected_price = scaler_y.inverse_transform([[y_scaled_pred]])[0][0]
    return projected_price