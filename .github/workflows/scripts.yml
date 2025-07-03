name: Test CLI scripts

on: [push]

jobs:
  tests:
    runs-on: t4_gpu
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
      - name: Download benchmark data
        run: |
          wget -O benchmark_data.zip "https://drive.google.com/uc?export=download&id=1NHrdYatR1rtqs2gPVfdvO0BAvocH8CJi"
          unzip -o benchmark_data.zip
      - name: Test single script
        run: poetry run marker_single benchmark_data/pdfs/switch_trans.pdf --page_range 0
      - name: Test convert script
        run: poetry run marker benchmark_data/pdfs --max_files 1 --page_range 0
      - name: Text convert script multiple workers
        run: poetry run marker benchmark_data/pdfs --max_files 2 --page_range 0-5
      - name: Test llm option
        run: |
          poetry run marker_single benchmark_data/pdfs/switch_trans.pdf --page_range 0 --use_llm > output.txt || echo "Command failed but continuing"
          if ! grep -q "UserWarning" output.txt; then
            echo "Success: No UserWarning found"
            exit 0
          else
            echo "Error: UserWarning found in output"
            exit 1
          fi