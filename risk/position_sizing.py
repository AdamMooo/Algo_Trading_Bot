def calculate_qty(cash_per_trade, price, volatility=None, is_crypto=False):
    """
    Calculate position size with volatility adjustment.
    
    Args:
        cash_per_trade: dollars you want to invest per trade
        price: current stock/crypto price
        volatility: optional volatility (std dev of returns) for adjustment
        is_crypto: if True, returns fractional quantity (crypto can be fractional)
        
    Returns: 
        number of shares/units to buy (int for stocks, float for crypto)
    """
    # Adjust cash for volatility -
    adjusted_cash = cash_per_trade
    if volatility is not None and volatility > 0:
        # Reduce position size for EXTREME volatility only
        # If vol = 0.02 (2%), multiplier = 1.0 (normal)
        # If vol = 0.10 (10%), multiplier = 0.8 (slight reduction)
        # Only cut to 75% minimum instead of 50%
        vol_multiplier = max(0.75, min(1.2, 0.02 / volatility))
        adjusted_cash = cash_per_trade * vol_multiplier
    
    # Calculate quantity
    if is_crypto:
        # Crypto can be fractional - use all available cash
        qty = adjusted_cash / price
        if qty < 0.001:  # Minimum crypto amount
            print(f"Warning: position too small ({qty:.6f} units)")
            return 0.001
        return round(qty, 6)  # Round to 6 decimals for crypto
    else:
        # Stocks must be whole shares
        qty = int(adjusted_cash // price)
        if qty < 1:
            print("Warning: not enough cash to buy even 1 share.")
            return 1  # Return 1 instead of 0 to allow small positions
        return qty

