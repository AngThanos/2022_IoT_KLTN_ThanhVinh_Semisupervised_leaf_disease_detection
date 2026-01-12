# Tests

This directory contains unit tests for the project.

## Structure

- Organize tests to mirror the structure of the `src/` directory
- Name test files with a `test_` prefix
- Group related tests into classes

## Best Practices

- Write tests for all new functionality
- Aim for high test coverage
- Use fixtures and mocks where appropriate
- Run tests regularly during development

## Running Tests

To run tests, use a testing framework like pytest:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src
```