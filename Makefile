
.PHONY: clean create_environment data evaluate
.SECONDARY: ## Save all intermediate files

#################################################################################
# GLOBALS                                                                       #
#################################################################################

RAW_DIR = data/raw
INTERIM_DIR = data/interim
PROCESSED_DIR = data/processed
RESULTS_DIR = results
DATASETS = $(PROCESSED_DIR)/DAVIS $(PROCESSED_DIR)/demo
RESULTS = $(RESULTS_DIR)/DAVIS/results.txt $(RESULTS_DIR)/demo/results.txt

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda env create -f environment.yml

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Prepare Video Object Removal Dataset
data: $(DATASETS)

$(PROCESSED_DIR)/% : $(PROCESSED_DIR)/%/InputImages $(PROCESSED_DIR)/%/Masks $(PROCESSED_DIR)/%/TargetImages ;

$(PROCESSED_DIR)/%/InputImages $(PROCESSED_DIR)/%/Masks $(PROCESSED_DIR)/%/TargetImages &: $(RAW_DIR)/%/JPEGImages $(INTERIM_DIR)/%/Masks
	python scripts/data/prepare_vor_dataset.py --images-dir $(word 1,$^) --masks-dir $(word 2,$^) --output-dir $(dir $@)

$(INTERIM_DIR)/%/Masks: $(RAW_DIR)/%/Annotations
	python scripts/data/masks/extract_masks.py --input-dir $(word 1,$^) --output-dir $@ --index 1

## Evaluate
evaluate: $(RESULTS)

$(RESULTS_DIR)/%/results.txt: $(RESULTS_DIR)/%/End2End/mean.txt $(RESULTS_DIR)/%/Inpainter/mean.txt $(RESULTS_DIR)/%/Tracker/mean.txt
	cat $^ > $@

$(RESULTS_DIR)/%/End2End/mean.txt: $(RESULTS_DIR)/%/End2End/OutputImages $(PROCESSED_DIR)/%/TargetImages
	python scripts/evaluate_inpainter.py --output-images-dir $(word 1,$^) --target-images-dir $(word 2,$^) --results-dir $(dir $@)

$(RESULTS_DIR)/%/Inpainter/mean.txt: $(RESULTS_DIR)/%/Inpainter/OutputImages $(PROCESSED_DIR)/%/TargetImages
	python scripts/evaluate_inpainter.py --output-images-dir $(word 1,$^) --target-images-dir $(word 2,$^) --results-dir $(dir $@)

$(RESULTS_DIR)/%/Tracker/mean.txt: $(RESULTS_DIR)/%/Tracker/OutputMasks $(PROCESSED_DIR)/%/Masks
	python scripts/evaluate_tracker.py --output-masks-dir $(word 1,$^) --target-masks-dir $(word 2,$^) --results-dir $(dir $@)

$(RESULTS_DIR)/%/End2End/OutputImages: $(PROCESSED_DIR)/%/InputImages $(RESULTS_DIR)/%/Tracker/OutputMasks
	python -W ignore::UserWarning scripts/infer_inpainter.py --input-images-dir $(word 1,$^) --input-masks-dir $(word 2,$^) --results-dir $(dir $@)

$(RESULTS_DIR)/%/Inpainter/OutputImages: $(PROCESSED_DIR)/%/InputImages $(PROCESSED_DIR)/%/Masks
	python -W ignore::UserWarning scripts/infer_inpainter.py --input-images-dir $(word 1,$^) --input-masks-dir $(word 2,$^) --results-dir $(dir $@)

$(RESULTS_DIR)/%/Tracker/OutputMasks: $(PROCESSED_DIR)/%/InputImages $(PROCESSED_DIR)/%/Masks
	python -W ignore::UserWarning scripts/infer_tracker.py --input-images-dir $(word 1,$^) --input-masks-dir $(word 2,$^) --results-dir $(dir $@)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}'
