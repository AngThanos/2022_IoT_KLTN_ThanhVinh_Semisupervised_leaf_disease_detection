# Code Style Guide

This document outlines the coding standards and style guidelines for this project. Following these guidelines ensures consistency across the codebase and makes it easier for contributors to understand and maintain the code.

## Python Style Guidelines

### General Principles

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code
- Use 4 spaces for indentation (no tabs)
- Keep line length to a maximum of 88 characters (compatible with Black formatter)
- Use meaningful variable and function names
- Write docstrings for all functions, classes, and modules

### Naming Conventions

- Use `snake_case` for variables, functions, and methods
- Use `PascalCase` for class names
- Use `UPPER_CASE` for constants
- Prefix private attributes and methods with a single underscore (`_`)

### Imports

- Group imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports
- Within each group, imports should be in alphabetical order
- Use absolute imports rather than relative imports

Example:
```python
# Standard library
import os
import sys
from datetime import datetime

# Third-party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local
from src.utils import data_loader
from src.models.model import Model
```

### Docstrings

Use [NumPy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for all functions, classes, and modules:

```python
def calculate_metric(predictions, targets, metric_type='accuracy'):
    """
    Calculate evaluation metric between predictions and targets.
    
    Parameters
    ----------
    predictions : array-like
        Model predictions
    targets : array-like
        Ground truth values
    metric_type : str, optional
        Type of metric to calculate, by default 'accuracy'
        
    Returns
    -------
    float
        Calculated metric value
        
    Raises
    ------
    ValueError
        If metric_type is not supported
    """
    # Function implementation
```

### Code Organization

- Keep functions and methods short and focused on a single task
- Use comments to explain complex logic or algorithms
- Separate logical sections of code with blank lines
- Use type hints where appropriate

## Code Quality Tools

This project uses the following tools to enforce code quality:

- [Black](https://black.readthedocs.io/) for code formatting
- [Flake8](https://flake8.pycqa.org/) for style guide enforcement
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](http://mypy-lang.org/) for static type checking

These tools are configured in the project's `.pre-commit-config.yaml` file and should be run before committing code.

## Jupyter Notebook Guidelines

- Keep notebooks clean and well-documented with markdown cells
- Move reusable code to Python modules in the `src/` directory
- Clear all outputs before committing notebooks
- Name notebooks with a prefix number for ordering (e.g., `01_data_exploration.ipynb`)

## Git Commit Messages

- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 50 characters
- Provide more detailed explanations in subsequent lines if necessary
- Reference issue numbers when relevant

Example:
```
Add data preprocessing module for time series

- Implement moving average smoothing
- Add outlier detection and removal
- Create data normalization functions

Fixes #42
```

## Enforcing the Style Guide

The style guide is enforced through:

1. Pre-commit hooks that run automated checks
2. Code review process
3. Continuous integration checks

Following these guidelines will help maintain a clean, consistent, and maintainable codebase.