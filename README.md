# Start Working On This Project

If you do not have uv installed on your device, please install it first:
```bash
pip install uv
```
To start working on this project, download project dependencies:
```bash
uv sync
```
This command will download project dependencies and build virtual environment for this project. All the dependencies required for this project are listed in the `pyproject.toml` file.

Then you can run the project with the command:
```bash
uv run main.py
```

# Project Structure
```
├── README.md