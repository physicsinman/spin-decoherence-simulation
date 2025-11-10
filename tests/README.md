# Test Suite for Spin Decoherence Simulation

This directory contains comprehensive tests for the simulation codebase.

## Test Structure

- `test_ornstein_uhlenbeck.py`: OU noise generation tests
- `test_coherence.py`: Phase accumulation and coherence function tests
- `test_noise_models.py`: Double-OU noise model tests
- `test_config.py`: Configuration validation tests
- `test_units.py`: Unit conversion helper tests

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_coherence.py -v
```

### Run specific test class
```bash
pytest tests/test_coherence.py::TestPhaseAccumulation -v
```

### Run specific test function
```bash
pytest tests/test_coherence.py::TestPhaseAccumulation::test_initial_condition -v
```

### Skip slow tests
```bash
pytest tests/ -v -m "not slow"
```

### Run only fast tests
```bash
pytest tests/ -v -m "not slow"
```

## Test Coverage

### Generate coverage report
```bash
pytest tests/ --cov=. --cov-report=html
```

This generates an HTML report in `htmlcov/index.html`.

### Coverage goals
- Unit tests: 80%+ coverage
- Integration tests: All major workflows
- Regression tests: Known bugs

## Test Categories

### Unit Tests
Fast, isolated tests for individual functions:
- `test_ornstein_uhlenbeck.py`
- `test_units.py`
- `test_config.py`

### Integration Tests
Tests for component interactions:
- `test_coherence.py` (ensemble coherence)
- `test_noise_models.py` (Double-OU)

### Regression Tests
Tests that prevent known bugs from reoccurring:
- All error handling tests
- Statistical property validation

## Markers

Tests are marked with:
- `@pytest.mark.slow`: Slow tests (skip with `-m "not slow"`)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.regression`: Regression tests
- `@pytest.mark.unit`: Unit tests

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest tests/ -v --cov=. --cov-report=xml
```

