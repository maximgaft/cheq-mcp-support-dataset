# Everything here runs offline with no API key.
UV := uv run python
STAGES := 01_load 02_clean 03_filter 04_dedup 05_split 06_embed 07_calibrate 08_database

.PHONY: label help fetch build eval serve check all
help:
	@echo "make fetch   download the three source CSVs (checksummed)"
	@echo "make build   run the 8-stage pipeline  -> data/interim/  (~80s)"
	@echo "make eval    routing eval + label agreement -> reports/  (~30s)"
	@echo "make serve   run the MCP server on stdio"
	@echo "make check   lint"
	@echo "make all     fetch + build + eval"

fetch:
	@./data/fetch.sh

build: fetch
	@$(foreach s,$(STAGES),echo "--- $(s)"; $(UV) pipeline/$(s).py || exit 1;)

eval:
	@$(UV) evals/run_routing.py
	@$(UV) evals/check_labels.py

serve:
	@$(UV) -m server

check:
	@uv run ruff check . --select F,E9,I

all: build eval

label:  ## classify what each first reply did (needs ANTHROPIC_API_KEY; gold set only)
	@uv run python pipeline/label.py

label-all:  ## same, but label the whole corpus (~$10) and write data/answer_labels.parquet
	@uv run python pipeline/label.py --all
