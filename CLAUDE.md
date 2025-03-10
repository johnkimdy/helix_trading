# Trading Codebase Guidelines

## Commands
- **Run code**: `python helix_alpha_gen.py`
- **Lint**: `flake8 --max-line-length=100 *.py`
- **Type check**: `mypy --ignore-missing-imports *.py`
- **Test**: `pytest` (add specific tests with `pytest test_file.py::test_function`)

## Code Style
- **Imports**: Group standard library, third-party, local imports with blank line between
- **Formatting**: 4-space indentation, 100 character line length
- **Typing**: Use type hints for function parameters and return values
- **Naming**:
  - Classes: PascalCase
  - Functions/methods: snake_case
  - Variables: snake_case
  - Constants: UPPER_SNAKE_CASE
- **Documentation**: Docstrings for all classes and functions
- **Error handling**: Use try/except blocks with specific exceptions
- **Threading**: Ensure clean thread shutdown and avoid race conditions

## Design Patterns
- Follow object-oriented principles with clear separation of concerns
- Use dependency injection for components like data providers