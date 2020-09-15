.PHONY: all install data tracking inpainting end2end
.SECONDARY: ## Save all intermediate files

#################################################################################
# CONFIG                                                                    #
#################################################################################

CONFIG = default.conf
include $(CONFIG)


#################################################################################
# GLOBALS                                                                       #
#################################################################################

DATA_DIR = data
DATA_RAW_DIR = $(DATA_DIR)/raw/$(DATASET)
DATA_INTERIM_DIR = $(DATA_DIR)/interim/$(DATASET)
DATA_PROCESSED_DIR = $(DATA_DIR)/processed/$(DATASET)

RESULTS_DIR = results
RESULTS_TRACKING_DIR = $(RESULTS_DIR)/tracking/$(DATASET)
RESULTS_INPAINTING_DIR = $(RESULTS_DIR)/inpainting/$(basename $(CONFIG))
RESULTS_END2END_DIR = $(RESULTS_DIR)/end2end/$(basename $(CONFIG))

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
install :
	conda env create -f environment.yml
	cd ./inpainting/external/layers/correlation_package && python setup.py install
	cd ./inpainting/external/layers/resample2d_package && python setup.py install
	cd ./inpainting/external/layers/channelnorm_package && python setup.py install

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## All
all : data tracking inpainting end2end

## Prepare Video Object Removal Dataset
data : $(DATA_PROCESSED_DIR)

$(DATA_PROCESSED_DIR) : $(DATA_PROCESSED_DIR)/InputImages $(DATA_PROCESSED_DIR)/Masks $(DATA_PROCESSED_DIR)/TargetImages ;

$(DATA_PROCESSED_DIR)/InputImages $(DATA_PROCESSED_DIR)/Masks $(DATA_PROCESSED_DIR)/TargetImages &: $(DATA_INTERIM_DIR)/AdjustedJPEGImages $(DATA_INTERIM_DIR)/AdjustedAnnotations $(DATA_INTERIM_DIR)/ObjectStats
	python scripts/data/prepare_vor_dataset.py --images-dir $(word 1,$^) \
                                               --annotations-dir $(word 2,$^) \
                                               --object-stats-dir $(word 3,$^) \
                                               --processed-dir $(dir $@) \
                                               --samples $(SAMPLES) \
                                               --seed $(SEED) \
                                               --min-presence $(MIN_PRESENCE) \
                                               --min-mean-size $(MIN_MEAN_SIZE) \
                                               --max-mean-size $(MAX_MEAN_SIZE)

$(DATA_INTERIM_DIR)/ObjectStats : $(DATA_INTERIM_DIR)/AdjustedAnnotations
	python scripts/data/calculate_object_stats.py --annotations-dir $(word 1,$^) \
                                                  --object-stats-dir $@

$(DATA_INTERIM_DIR)/AdjustedAnnotations : $(DATA_RAW_DIR)/Annotations
	python scripts/data/adjust_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --crop $(CROP_WIDTH) $(CROP_HEIGHT) \
                                         --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
                                         --frame-type 'annotation'

$(DATA_INTERIM_DIR)/AdjustedJPEGImages: $(DATA_RAW_DIR)/JPEGImages
	python scripts/data/adjust_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --crop $(CROP_WIDTH) $(CROP_HEIGHT) \
                                         --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
                                         --frame-type 'image'


## Tracking
tracking : $(RESULTS_TRACKING_DIR)

$(RESULTS_TRACKING_DIR) : $(RESULTS_TRACKING_DIR)/Evaluation $(RESULTS_TRACKING_DIR)/Initialization $(RESULTS_TRACKING_DIR)/OutputMasks

$(RESULTS_TRACKING_DIR)/Evaluation : $(RESULTS_TRACKING_DIR)/OutputMasks $(DATA_PROCESSED_DIR)/Masks
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode segmentation

$(RESULTS_TRACKING_DIR)/Initialization $(RESULTS_TRACKING_DIR)/OutputMasks &: $(DATA_PROCESSED_DIR)/InputImages $(DATA_PROCESSED_DIR)/Masks
	python scripts/infer_tracking.py --input-images-dir $(word 1,$^) \
                                     --input-masks-dir $(word 2,$^) \
                                     --results-dir $(dir $@)

## Inpainting
inpainting : $(RESULTS_INPAINTING_DIR)

$(RESULTS_INPAINTING_DIR) : $(RESULTS_INPAINTING_DIR)/Evaluation $(RESULTS_INPAINTING_DIR)/OutputImages

$(RESULTS_INPAINTING_DIR)/Evaluation: $(RESULTS_INPAINTING_DIR)/OutputImages $(DATA_PROCESSED_DIR)/TargetImages
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode inpainting

$(RESULTS_INPAINTING_DIR)/OutputImages: $(DATA_PROCESSED_DIR)/InputImages $(DATA_PROCESSED_DIR)/Masks
	python scripts/infer_inpainting.py --input-images-dir $(word 1,$^) \
                                       --input-masks-dir $(word 2,$^) \
                                       --results-dir $(dir $@)


## End2End

end2end : $(RESULTS_END2END_DIR)

$(RESULTS_END2END_DIR) : $(RESULTS_END2END_DIR)/Evaluation $(RESULTS_END2END_DIR)/OutputImages

$(RESULTS_END2END_DIR)/Evaluation: $(RESULTS_END2END_DIR)/OutputImages $(DATA_PROCESSED_DIR)/TargetImages
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode inpainting

$(RESULTS_END2END_DIR)/OutputImages: $(DATA_PROCESSED_DIR)/InputImages $(RESULTS_TRACKING_DIR)/OutputMasks
	python scripts/infer_inpainting.py --input-images-dir $(word 1,$^) \
                                       --input-masks-dir $(word 2,$^) \
                                       --results-dir $(dir $@)
