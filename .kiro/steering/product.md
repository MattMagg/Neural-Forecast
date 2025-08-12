# Product Overview

## Core Purpose
Intraday Bitcoin (BTC) forecasting system using 15-minute bars with calibrated prediction intervals. The system provides probabilistic forecasts for multiple horizons (1h, 2h, 4h, 8h) with uncertainty quantification.

## Key Features
- **Target**: Log returns on 15-minute BTC price data
- **Horizons**: 4, 8, 16, 32 steps (1h, 2h, 4h, 8h ahead)
- **Prediction Intervals**: 80%, 90%, 95% coverage targets (±2% tolerance)
- **Primary Metric**: sCRPS (Scaled Continuous Ranked Probability Score)
- **Architecture**: NeuralForecast-centric approach with no custom implementations

## Technical Constraints
- **Data Frequency**: 15-minute bars with UTC end-of-bar timestamps
- **Feature Limit**: Maximum 256 features after pruning
- **Leakage Prevention**: All historical features must be shifted by 1 bar
- **Grid Requirements**: Regular time grid with no gaps
- **Model Portfolio**: NHITS, NBEATSx, TiDE, PatchTST with probabilistic losses

## Success Criteria
- Coverage targets within ±2% tolerance for all prediction intervals
- Competitive sCRPS scores across all horizons
- Production-ready inference pipeline with sub-second latency
