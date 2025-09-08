# Paperspace Gradient Environment for AutoGluon + Hugging Face + VectorBT

[![Docker Build](https://github.com/joephaser/paperspaceenvironment/actions/workflows/docker-build.yml/badge.svg)](https://github.com/joephaser/paperspaceenvironment/actions/workflows/docker-build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Pulls](https://img.shields.io/docker/pulls/ghcr.io/joephaser/paperspaceenvironment)](https://github.com/joephaser/paperspaceenvironment/pkgs/container/paperspaceenvironment)

This Docker environment provides a complete setup for time series analysis, machine learning, and financial backtesting on Paperspace Gradient.

## Features

- **PyTorch 2.1.0 with CUDA 11.8 + cuDNN 8** for GPU acceleration
- **AutoGluon** for time series prediction and tabular ML
- **Hugging Face Transformers** for running pre-trained models
- **VectorBT** for financial backtesting and analysis
- **JupyterLab** pre-configured for Paperspace
- **Comprehensive data science stack** (pandas, numpy, matplotlib, etc.)

## Quick Start

### Using Pre-built Image (Recommended)

The easiest way to get started is using our pre-built image from GitHub Container Registry:

```bash
# Pull the latest image
docker pull ghcr.io/joephaser/paperspaceenvironment:latest

# Run locally with GPU support
docker run -it --gpus all -p 8888:8888 ghcr.io/joephaser/paperspaceenvironment:latest

# Start JupyterLab
start-jupyter.sh
```

> **Note**: Images are built using GitHub Actions with optimized caching and space management to handle the large ML dependencies.

### Building from Source

```bash
# Clone the repository
git clone https://github.com/joephaser/paperspaceenvironment.git
cd paperspaceenvironment

# Build the image (requires significant disk space)
docker build -t paperspace-autogluon .

# Or use the multi-stage build for smaller final image
docker build -f Dockerfile.multistage -t paperspace-autogluon .

# Run the container
docker run -it --gpus all -p 8888:8888 paperspace-autogluon
```

> **Warning**: Local builds require ~15-20GB of disk space due to ML dependencies. Consider using the pre-built image instead.

### Using Docker Compose (Development)

For local development with persistent volumes:

```bash
# Clone and start with Docker Compose
git clone https://github.com/joephaser/paperspaceenvironment.git
cd paperspaceenvironment

# Start the environment
docker-compose up

# Access JupyterLab at http://localhost:8888
```

This will:
- Build the image if needed
- Mount `./notebooks` and `./data` directories for persistent storage
- Start JupyterLab automatically

### Using on Paperspace Gradient

1. Create a new Gradient notebook
2. Use the custom container: `ghcr.io/joephaser/paperspaceenvironment:latest`
3. JupyterLab will be available automatically
4. Or use the Paperspace CLI:

```bash
gradient notebooks create \
  --name "AutoGluon Environment" \
  --projectId your-project-id \
  --machineType P4000 \
  --container ghcr.io/joephaser/paperspaceenvironment:latest
```

## Included Packages

### Core ML & Data Science
- AutoGluon (full suite including timeseries)
- scikit-learn
- pandas, numpy, scipy
- matplotlib, seaborn, plotly

### Hugging Face Ecosystem
- transformers
- datasets
- accelerate
- tokenizers
- safetensors
- bitsandbytes (for quantization)

### Financial Analysis
- VectorBT (backtesting framework)
- yfinance (market data)
- QuantLib (quantitative finance)
- TA-Lib (technical analysis indicators)

### Time Series Specific
- statsmodels
- pmdarima (auto-ARIMA)

## Environment Configuration

### CUDA/GPU
The environment is configured for CUDA 11.8 with PyTorch 2.1.0. Ensure your Paperspace machine has compatible drivers.

### Hugging Face
- Models cache to `/home/tradelab/.cache/huggingface/`
- Telemetry disabled
- Hub authentication ready (set `HF_TOKEN` if needed)

### JupyterLab
- Runs on port 8888
- No authentication required (suitable for Paperspace)
- Extensions pre-installed

## Sample Usage

### AutoGluon Time Series
```python
from autogluon.timeseries import TimeSeriesPredictor
import pandas as pd

# Load your time series data
df = pd.read_csv('your_timeseries.csv')
predictor = TimeSeriesPredictor(prediction_length=24)
predictor.fit(df)
predictions = predictor.predict(df)
```

### VectorBT Backtesting
```python
import vectorbt as vbt
import yfinance as yf

# Get data and run backtest
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
entries = data['Close'].rolling(50).mean() > data['Close'].rolling(200).mean()
portfolio = vbt.Portfolio.from_signals(data['Close'], entries, exits=~entries)
print(portfolio.stats())
```

### TA-Lib Technical Analysis
```python
import talib
import numpy as np

# Calculate technical indicators
close_prices = np.array([100, 102, 101, 103, 105, 104, 106])
sma = talib.SMA(close_prices, timeperiod=5)
rsi = talib.RSI(close_prices, timeperiod=5)
macd, signal, hist = talib.MACD(close_prices)

print(f"Available indicators: {len(talib.get_functions())}")
```

### Hugging Face Models
```python
from transformers import pipeline

# Use a pre-trained model
classifier = pipeline("sentiment-analysis")
result = classifier("The market outlook is positive!")
print(result)
```

## Testing Your Environment

Run the included test notebook:
```bash
jupyter lab test_environment.ipynb
```

This will verify all major components are working correctly.

## Troubleshooting

### GPU Issues
- Check CUDA availability: `torch.cuda.is_available()`
- Verify driver compatibility with CUDA 11.8

### Memory Issues
- Monitor GPU memory: `nvidia-smi`
- Use gradient checkpointing for large models
- Consider using bitsandbytes for model quantization

### Package Conflicts
- The environment uses latest stable versions
- For specific versions, modify the Dockerfile pip install commands

### Build Issues (Local Development)
- **Disk Space**: Local builds require ~15-20GB of free space
- **Memory**: Recommend 16GB+ RAM for building
- **Solution**: Use pre-built images from GitHub Container Registry instead

## Docker Image Optimization

This repository uses several optimizations to manage the large size of ML dependencies:

### GitHub Actions Optimizations
- **Remote Building**: Uses Docker Buildx to build remotely and push directly to registry
- **Disk Cleanup**: Frees up GitHub runner disk space before building
- **Layer Caching**: Aggressive caching to speed up subsequent builds
- **Combined RUN Commands**: Reduces intermediate layer size

### Dockerfile Optimizations
- **Single RUN Layer**: All pip installs combined into one layer
- **Cache Purging**: Removes pip cache after installation
- **Minimal Base**: PyTorch base image instead of full CUDA image
- **Multi-stage Option**: Available in `Dockerfile.multistage` for even smaller images

## Customization

To add additional packages, modify the Dockerfile:

```dockerfile
RUN python3 -m pip install --no-cache-dir \
    your-additional-package
```

For system dependencies:
```dockerfile
RUN apt-get update && apt-get install -y \
    your-system-package \
    && rm -rf /var/lib/apt/lists/*
```

## License

This environment configuration is provided under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Reporting Issues

If you encounter any issues or have suggestions for improvements, please [open an issue](https://github.com/joephaser/paperspaceenvironment/issues) on GitHub.

## Acknowledgments

- [AutoGluon](https://auto.gluon.ai/) for the amazing AutoML framework
- [Hugging Face](https://huggingface.co/) for the transformers library
- [VectorBT](https://vectorbt.dev/) for backtesting capabilities
- [Paperspace](https://www.paperspace.com/) for the ML platform

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=joephaser/paperspaceenvironment&type=Date)](https://star-history.com/#joephaser/paperspaceenvironment&Date)
