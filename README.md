# Topic Sorting Project

This is a Python project for topic sorting.

## Project Structure

```
topic-sorting/
├── assets/              # Data files (JSON, etc.)
├── src/                 # Source code
│   └── topic_sorting/   # Main package
│       ├── __init__.py
│       └── main.py      # Main entry point
├── tests/               # Test files
├── venv/               # Virtual environment
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd topic-sorting
```

2. Create and activate virtual environment:

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Development

- Use `black` for code formatting
- Use `flake8` for linting
- Use `pytest` for testing

## Running the Project

To run the main script:

```bash
python -m src.main
```

## License

[Add your license here]
