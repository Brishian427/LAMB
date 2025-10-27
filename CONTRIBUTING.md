# Contributing to LAMB

We welcome contributions to LAMB! This document provides guidelines for contributing to the OASIS-Fudan Complex System AI Social Scientist Team project.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Installation

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/lamb.git
   cd lamb
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   pip install -e .[dev]  # Install development dependencies
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black formatter)
- Use type hints for all function parameters and return values
- Use docstrings for all public functions and classes
- Follow NumPy docstring format for scientific functions

### Code Formatting

We use Black for code formatting and isort for import sorting:

```bash
black lamb/
isort lamb/
```

### Type Checking

We use mypy for static type checking:

```bash
mypy lamb/
```

### Linting

We use flake8 for linting:

```bash
flake8 lamb/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lamb

# Run specific test file
pytest tests/test_basic_functionality.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Aim for high test coverage (>90%)
- Use descriptive test names
- Include both positive and negative test cases
- Test edge cases and error conditions

### Test Structure

```python
def test_function_name():
    """Test description explaining what is being tested."""
    # Arrange
    setup_test_data()
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected_value
```

## Documentation

### Docstring Format

Use NumPy-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Brief description of the function.
    
    More detailed description if needed.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
        
    Returns
    -------
    bool
        Description of return value
        
    Raises
    ------
    ValueError
        When param1 is negative
        
    Examples
    --------
    >>> function_name(1, "test")
    True
    """
```

### Documentation Building

```bash
# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Run code formatting and linting
3. Update documentation if needed
4. Add tests for new functionality
5. Update CHANGELOG.md

### Pull Request Guidelines

1. **Title**: Use clear, descriptive title
2. **Description**: Explain what changes were made and why
3. **Tests**: Ensure all tests pass
4. **Documentation**: Update relevant documentation
5. **Breaking Changes**: Clearly mark any breaking changes

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] All existing tests still pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, LAMB version
6. **Code Sample**: Minimal code to reproduce the issue

### Feature Requests

When requesting features, please include:

1. **Description**: Clear description of the feature
2. **Use Case**: Why this feature would be useful
3. **Proposed Solution**: How you think it should work
4. **Alternatives**: Other solutions you've considered

## Code of Conduct

### Our Pledge

We, the OASIS-Fudan Complex System AI Social Scientist Team, are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or inappropriate comments
- Personal attacks or political discussions
- Public or private harassment
- Publishing private information without permission
- Other unprofessional conduct

## Development Workflow

### Branch Naming

- `feature/description`: New features
- `bugfix/description`: Bug fixes
- `docs/description`: Documentation updates
- `refactor/description`: Code refactoring

### Commit Messages

Use clear, descriptive commit messages:

```
Add support for custom agent behaviors

- Implement CustomAgent base class
- Add behavior configuration system
- Update documentation with examples
- Add tests for new functionality
```

### Release Process

1. Update version in `__version__.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and upload to PyPI

## Getting Help

- **Documentation**: Check the docs first
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers directly for sensitive issues

## Recognition

Contributors will be recognized in:

- AUTHORS.md file
- Release notes
- Project documentation
- Annual contributor acknowledgments

Thank you for contributing to LAMB!
