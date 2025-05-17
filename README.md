# NBA Team Classifier

A deep learning model that classifies NBA team logos using a custom ResNet-style architecture.

## Project Description

This project implements a convolutional neural network (CNN) called EvenBetterNet that can classify NBA team logos. The model uses residual blocks for better feature extraction and gradient flow.

## Model Architecture

The model consists of:
- 5 Residual blocks with increasing channel dimensions (32 → 512)
- Adaptive Average Pooling
- Fully connected layers with dropout for classification
- Output layer for 30 NBA teams

## Requirements

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
NBATeamClassifier/
├── models/
│   ├── evenbetternet.py    # Main model architecture
│   └── residualblocks.py   # Residual block implementation
├── requirements.txt
└── README.md
```

## Usage

```python
from models.evenbetternet import EvenBetterNet
import torch

# Initialize the model
model = EvenBetterNet()

# Load an image (3 channels, Height, Width)
# image = ... # Your image loading code here

# Make prediction
with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1)
```

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```