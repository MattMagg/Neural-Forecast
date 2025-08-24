# Bitcoin Forecasting System

A production-ready probabilistic forecasting system for Bitcoin price prediction using 15-minute intervals with calibrated prediction intervals.

## Overview

This system delivers high-quality probabilistic forecasts for BTC at 15-minute frequency with calibrated 80%/90%/95% prediction intervals across multiple time horizons (1h, 2h, 4h, 8h). Built with a **NeuralForecast-centric** approach, the system emphasizes data discipline, leakage prevention, and production readiness.

## Key Features

### Forecasting Capabilities

- **Multi-horizon predictions**: 1h, 2h, 4h, 8h ahead forecasts
- **Probabilistic outputs**: Calibrated prediction intervals at 80%, 90%, 95% confidence levels
- **Primary metric**: sCRPS (Scaled Continuous Ranked Probability Score)
- **Target variable**: Log returns on 15-minute BTC price data

### Technical Architecture

- **NeuralForecast-native**: Uses NF primitives for modeling, cross-validation, and persistence
- **Model portfolio**: NHITS, NBEATSx, TiDE, PatchTST with probabilistic losses
- **Feature engineering**: Technical indicators with multi-timeframe analysis (≤256 features)
- **Data discipline**: UTC timestamps, regular 15-minute grid, strict leakage prevention

## Project Structure

```
├── data/                   # Raw and processed datasets
├── features/               # Feature engineering pipeline
├── nf_models/              # Model factory and configurations
├── cv/                     # Cross-validation and metrics
├── uq/                     # Uncertainty quantification
├── experiments/            # Per-horizon configurations and results
├── utils/                  # Validation and I/O utilities
├── tests/                  # Comprehensive test suite
└── examples/               # Usage examples and tutorials
```

## Core Components

### Data Processing

- **Input**: 1-minute OHLCV Bitcoin data from Kaggle
- **Output**: Regular 15-minute UTC grid with log returns
- **Validation**: Comprehensive quality gates preventing data leakage

### Feature Engineering

- **Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
- **Multi-timeframe**: 30min, 1h, 4h aligned to 15min base
- **Leakage prevention**: Strict shift(1) rule for historical features

### Model Training

- **Cross-validation**: NeuralForecast-native with proper windowing
- **Model selection**: sCRPS-based ranking with ensemble options
- **Persistence**: NF-native save/load with versioning

### Uncertainty Quantification

- **Coverage diagnostics**: Empirical coverage validation
- **PIT analysis**: Probability integral transform uniformity testing
- **Conformal prediction**: NF's PredictionIntervals integration

## Acknowledgments

Built with [NeuralForecast](https://github.com/Nixtla/neuralforecast) by Nixtla for state-of-the-art neural forecasting capabilities.

---

*For detailed technical information, see the [Technical Specification](docs/forecasting_sf_plan.md).*
