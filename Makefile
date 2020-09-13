
.PHONY: clean create_environment data evaluate demo install
.SECONDARY: ## Save all intermediate files

#################################################################################
# PARAMETERS                                                                    #
#################################################################################

HEIGHT = 256
WIDTH = 512
MIN_PRESENCE = 0.75
MIN_MEAN_SIZE = 0.10
MAX_MEAN_SIZE = 0.25

#################################################################################
# GLOBALS                                                                       #
#################################################################################

RAW_DIR = data/raw
INTERIM_DIR = data/interim
PROCESSED_DIR = data/processed
RESULTS_DIR = results
DATASETS = $(PROCESSED_DIR)/DAVIS
RESULTS = $(RESULTS_DIR)/DAVIS
DEMO = $(PROCESSED_DIR)/demo $(RESULTS_DIR)/demo

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda env create -f environment.yml

## Install layers
install:
	cd ./inpainting/external/layers/correlation_package && python setup.py install
	cd ./inpainting/external/layers/resample2d_package && python setup.py install
	cd ./inpainting/external/layers/channelnorm_package && python setup.py install

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Demo data and evaluation
demo: $(DEMO)

## Prepare Video Object Removal Dataset
data: $(DATASETS)

$(PROCESSED_DIR)/% : $(PROCESSED_DIR)/%/InputImages $(PROCESSED_DIR)/%/Masks $(PROCESSED_DIR)/%/TargetImages ;

$(PROCESSED_DIR)/%/InputImages $(PROCESSED_DIR)/%/Masks $(PROCESSED_DIR)/%/TargetImages &: $(INTERIM_DIR)/%/ResizedJPEGImages $(INTERIM_DIR)/%/ResizedAnnotations $(INTERIM_DIR)/%/ObjectStats
	python scripts/data/prepare_vor_dataset.py --images-dir $(word 1,$^) \
                                               --annotations-dir $(word 2,$^) \
                                               --object-stats-dir $(word 3,$^) \
                                               --processed-dir $(dir $@) \
                                               --min-presence $(MIN_PRESENCE) \
                                               --min-mean-size $(MIN_MEAN_SIZE) \
                                               --max-mean-size $(MAX_MEAN_SIZE)

$(INTERIM_DIR)/%/ObjectStats: $(INTERIM_DIR)/%/ResizedAnnotations
	python scripts/data/calculate_object_stats.py --annotations-dir $(word 1,$^) \
                                                  --object-stats-dir $@

$(INTERIM_DIR)/%/ResizedAnnotations: $(RAW_DIR)/%/Annotations
	python scripts/data/resize_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --size $(WIDTH) $(HEIGHT) \
                                         --frame-type 'annotation'

$(INTERIM_DIR)/%/ResizedJPEGImages: $(RAW_DIR)/%/JPEGImages
	python scripts/data/resize_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --size $(WIDTH) $(HEIGHT) \
                                         --frame-type 'image'


## Evaluate
evaluate: $(RESULTS)

$(RESULTS_DIR)/% : $(RESULTS_DIR)/%/End2End/Evaluation $(RESULTS_DIR)/%/Inpainter/Evaluation $(RESULTS_DIR)/%/Tracker/Evaluation ;

$(RESULTS_DIR)/%/End2End/Evaluation: $(RESULTS_DIR)/%/End2End/OutputImages $(PROCESSED_DIR)/%/TargetImages
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode inpainting

$(RESULTS_DIR)/%/Inpainter/Evaluation: $(RESULTS_DIR)/%/Inpainter/OutputImages $(PROCESSED_DIR)/%/TargetImages
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode inpainting

$(RESULTS_DIR)/%/Tracker/Evaluation: $(RESULTS_DIR)/%/Tracker/OutputMasks $(PROCESSED_DIR)/%/Masks
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode segmentation

$(RESULTS_DIR)/%/End2End/OutputImages: $(PROCESSED_DIR)/%/InputImages $(RESULTS_DIR)/%/Tracker/OutputMasks
	python scripts/infer_inpainting.py --input-images-dir $(word 1,$^) \
									   --input-masks-dir $(word 2,$^) \
									   --results-dir $(dir $@)

$(RESULTS_DIR)/%/Inpainter/OutputImages: $(PROCESSED_DIR)/%/InputImages $(PROCESSED_DIR)/%/Masks
	python scripts/infer_inpainting.py --input-images-dir $(word 1,$^) \
                                       --input-masks-dir $(word 2,$^) \
                                       --results-dir $(dir $@)

$(RESULTS_DIR)/%/Tracker/OutputMasks: $(PROCESSED_DIR)/%/InputImages $(PROCESSED_DIR)/%/Masks
	python scripts/infer_tracking.py --input-images-dir $(word 1,$^) \
                                     --input-masks-dir $(word 2,$^) \
                                     --results-dir $(dir $@)
