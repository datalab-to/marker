name: Python package
on:
  push:
    tags:
      - "v*.*.*"
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: 3.11
      - name: Install python dependencies
        run: |
          pip install poetry
          poetry install --extras "full"
      - name: Build package
        run: |
          poetry build
      - name: Extract version from pyproject.toml
        id: version
        run: |
          VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['tool']['poetry']['version'])")
          echo "version=v$VERSION" >> $GITHUB_OUTPUT
      - name: Publish package
        env:
          PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
        run: |
          poetry config pypi-token.pypi "$PYPI_TOKEN"
          poetry publish
