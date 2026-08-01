# LAB BTC vs Alts Divergence

Research branch for reconstructing Binance Spot BTC/ETH/SOL/ADA/XRP/DOGE/LTC data, resampling it causally, and studying candle-by-candle divergence, relative volume, breadth, and the supplied stateful ADX Dominante regime.

V2 separates pairwise histories (ADA/BTC starts with ADA availability) from six-alt breadth (starts when all six alts overlap) and uses prior completed candles for the volume baseline.
