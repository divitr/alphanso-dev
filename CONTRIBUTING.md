# Contributing to ALPHANSO

Thank you for your interest in contributing to ALPHANSO!

## Getting Started

### Prerequisites

- Python >= 3.10
- Dependencies: numpy, pandas, scipy, PyYAML

### Development Setup

```bash
git clone https://github.com/alphanso-org/alphanso.git
cd alphanso
pip install -e .
```

### Running Tests

```bash
pytest
```

No special setup or external services are required.

## How to Contribute

### Reporting Bugs or Requesting Features

Please [open an issue](https://github.com/alphanso-org/alphanso/issues) before starting work. This helps avoid duplicate effort and ensures alignment on the approach.

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests if applicable
4. Run the test suite to confirm nothing is broken (`pytest`)
5. Commit your changes and push to your fork
6. Open a pull request against `main`

### Pull Request Guidelines

- Keep PRs focused on a single change
- Include a clear description of what the PR does and why
- Ensure all tests pass before requesting review

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this standard. Please report concerns via [GitHub Issues](https://github.com/alphanso-org/alphanso/issues) or by contacting the maintainers directly.
