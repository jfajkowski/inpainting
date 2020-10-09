.PHONY: install data videos ts ii fe fi ee
.SECONDARY: ## Save all intermediate files

#################################################################################
# CONFIG                                                                        #
#################################################################################

include default.conf
CONFIG = default.conf
include $(CONFIG)

#################################################################################
# DIRECTORIES                                                                   #
#################################################################################

VIDEO_DATASET = demo
FLOW_DATASET = demo

DATA_DIR = data
DATA_RAW_DIR = $(DATA_DIR)/raw
DATA_INTERIM_DIR = $(DATA_DIR)/interim
DATA_PROCESSED_DIR = $(DATA_DIR)/processed

DATA_FE_DIR = $(DATA_PROCESSED_DIR)/fe/$(FLOW_DATASET)
DATA_FI_DIR = $(DATA_PROCESSED_DIR)/fi/$(FLOW_DATASET)
DATA_II_DIR = $(DATA_PROCESSED_DIR)/ii/$(VIDEO_DATASET)
DATA_TS_DIR = $(DATA_PROCESSED_DIR)/ts/$(VIDEO_DATASET)
DATA_EE_DIR = $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)

RESULTS_DIR = results
RESULTS_FE_DIR = $(RESULTS_DIR)/fe/$(FLOW_DATASET)/$(basename $(CONFIG))
RESULTS_FI_DIR = $(RESULTS_DIR)/fi/$(FLOW_DATASET)/$(basename $(CONFIG))
RESULTS_II_DIR = $(RESULTS_DIR)/ii/$(VIDEO_DATASET)/$(basename $(CONFIG))
RESULTS_TS_DIR = $(RESULTS_DIR)/ts/$(VIDEO_DATASET)/$(basename $(CONFIG))
RESULTS_EE_DIR = $(RESULTS_DIR)/ee/$(VIDEO_DATASET)/$(basename $(CONFIG))

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
install :
	conda env create -f environment.yml
	cd ./inpainting/external/layers/correlation_package && python setup.py install
	cd ./inpainting/external/layers/resample2d_package && python setup.py install
	cd ./inpainting/external/layers/channelnorm_package && python setup.py install

## Prepare data
data : $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/Annotations \
       $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/Images \
       $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/ObjectStats \
       $(DATA_INTERIM_DIR)/$(FLOW_DATASET)/Images \
       $(DATA_INTERIM_DIR)/$(FLOW_DATASET)/Flows


$(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/Annotations : $(DATA_RAW_DIR)/$(VIDEO_DATASET)/Annotations
	python scripts/data/adjust_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --crop $(CROP_RATIO) \
                                         --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
                                         --frame-type 'annotation'

$(DATA_INTERIM_DIR)/$(FLOW_DATASET)/Flows : $(DATA_RAW_DIR)/$(FLOW_DATASET)/Flows
	python scripts/data/adjust_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --crop $(CROP_RATIO) \
                                         --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
                                         --frame-type 'flow'

$(DATA_INTERIM_DIR)/%/Images : $(DATA_RAW_DIR)/%/Images
	python scripts/data/adjust_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --crop $(CROP_RATIO) \
                                         --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
                                         --frame-type 'image'

$(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/ObjectStats : $(DATA_RAW_DIR)/$(VIDEO_DATASET)/Annotations
	python scripts/data/calculate_object_stats.py --annotations-dir $(word 1,$^) \
                                                  --object-stats-dir $@ \
												  --min-presence $(MIN_PRESENCE) \
												  --min-mean-size $(MIN_MEAN_SIZE) \
												  --max-mean-size $(MAX_MEAN_SIZE)

## Prepare videos
videos : $(DATA_TS_DIR)/ImageVideos $(DATA_TS_DIR)/MaskVideos $(RESULTS_TS_DIR)/MaskVideos \
         $(DATA_II_DIR)/ImageVideos $(DATA_II_DIR)/MaskVideos $(RESULTS_II_DIR)/ImageVideos \
         $(DATA_FE_DIR)/ImageVideos $(DATA_FE_DIR)/FlowVideos $(RESULTS_FE_DIR)/FlowVideos \
         $(DATA_FI_DIR)/FlowVideos $(DATA_FI_DIR)/MaskVideos $(RESULTS_FI_DIR)/FlowVideos \
         $(DATA_EE_DIR)/ImageVideos $(RESULTS_EE_DIR)/ImageVideos $(RESULTS_EE_DIR)/MaskVideos


%/FlowVideos : %/Flows
	python scripts/convert_to_videos.py --frames-dir $^ \
	                                    --videos-dir $@ \
	                                    --frame-type 'flow'

%/ImageVideos : %/Images
	python scripts/convert_to_videos.py --frames-dir $^ \
	                                    --videos-dir $@ \
	                                    --frame-type 'image'
%/MaskVideos : %/Masks
	python scripts/convert_to_videos.py --frames-dir $^ \
	                                    --videos-dir $@ \
	                                    --frame-type 'mask'

#################################################################################
# EXPERIMENTS                                                                   #
#################################################################################

## Tracking and segmentation
ts : $(RESULTS_TS_DIR)/Benchmark \
	 $(RESULTS_TS_DIR)/Evaluation \
	 $(RESULTS_TS_DIR)/Initialization \
	 $(RESULTS_TS_DIR)/Masks

$(RESULTS_TS_DIR)/Evaluation : $(RESULTS_TS_DIR)/Masks \
                               $(DATA_TS_DIR)/Masks
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode 'tracking_and_segmentation'

$(RESULTS_TS_DIR)/Benchmark $(RESULTS_TS_DIR)/Initialization $(RESULTS_TS_DIR)/Masks &: $(DATA_TS_DIR)/Images \
                                                                                        $(DATA_TS_DIR)/Masks
	python scripts/infer_tracking_and_segmentation.py --images-dir $(word 1,$^) \
													  --masks-dir $(word 2,$^) \
													  --results-dir $(dir $@) \
													  --dilation-size $(DILATION_SIZE) \
													  --dilation-iterations $(DILATION_ITERATIONS)


$(DATA_TS_DIR)/Images $(DATA_TS_DIR)/Masks &: $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/Images \
											  $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/Annotations \
											  $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/ObjectStats
	python scripts/data/prepare_dataset.py --frames-dir $(word 1,$^) \
										   --annotations-dir $(word 2,$^) \
										   --object-stats-dir $(word 3,$^) \
										   --processed-dir $(dir $@) \
										   --limit-samples $(SAMPLES) \
										   --seed $(SEED) \
										   --frame-type 'image' \
										   --mode 'match'

## Image inpainting
ii : $(RESULTS_II_DIR)/Benchmark \
	 $(RESULTS_II_DIR)/Evaluation \
     $(RESULTS_II_DIR)/Images

$(RESULTS_II_DIR)/Evaluation : $(RESULTS_II_DIR)/Images \
                               $(DATA_II_DIR)/Images
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode 'image_inpainting'

$(RESULTS_II_DIR)/Benchmark $(RESULTS_II_DIR)/Images &: $(DATA_II_DIR)/Images \
                                                        $(DATA_II_DIR)/Masks
	python scripts/infer_inpainting.py --frames-dir $(word 1,$^) \
                                       --masks-dir $(word 2,$^) \
                                       --results-dir $(dir $@) \
                                       --inpainting-model $(IMAGE_INPAINTING_MODEL) \
                                       --flow-model $(FLOW_MODEL) \
                                       --mode 'image_inpainting'

$(DATA_II_DIR)/Images $(DATA_II_DIR)/Masks &: $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/Images \
											  $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/Annotations \
											  $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/ObjectStats
	python scripts/data/prepare_dataset.py --frames-dir $(word 1,$^) \
										   --annotations-dir $(word 2,$^) \
										   --object-stats-dir $(word 3,$^) \
										   --processed-dir $(dir $@) \
										   --limit-samples $(SAMPLES) \
										   --seed $(SEED) \
										   --frame-type 'image' \
										   --mode 'cross'

## Flow estimation
fe : $(RESULTS_FE_DIR)/Benchmark \
     $(RESULTS_FE_DIR)/Evaluation \
     $(RESULTS_FE_DIR)/Flows

$(RESULTS_FE_DIR)/Evaluation : $(RESULTS_FE_DIR)/Flows \
                               $(DATA_FE_DIR)/Flows
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode 'flow_estimation'

$(RESULTS_FE_DIR)/Benchmark $(RESULTS_FE_DIR)/Flows &: $(DATA_FE_DIR)/Images
	python scripts/infer_flow_estimation.py --images-dir $(word 1,$^) \
                                            --results-dir $(dir $@) \
                                            --flow-model $(FLOW_MODEL)

$(DATA_FE_DIR)/Images $(DATA_FE_DIR)/Flows &: $(DATA_INTERIM_DIR)/$(FLOW_DATASET)/Images \
                                              $(DATA_INTERIM_DIR)/$(FLOW_DATASET)/Flows
	python scripts/data/prepare_flow_estimation_dataset.py --images-dir $(word 1,$^) \
										   				   --flows-dir $(word 2,$^) \
													       --processed-dir $(dir $@) \
													       --limit-samples $(SAMPLES) \
													       --seed $(SEED)

## Flow inpainting
fi : $(RESULTS_FI_DIR)/Benchmark \
	 $(RESULTS_FI_DIR)/Evaluation \
     $(RESULTS_FI_DIR)/Flows

$(RESULTS_FI_DIR)/Evaluation : $(RESULTS_FI_DIR)/Flows \
                               $(DATA_FI_DIR)/Flows
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode 'flow_inpainting'

$(RESULTS_FI_DIR)/Benchmark $(RESULTS_FI_DIR)/Flows &: $(DATA_FI_DIR)/Flows \
                                                       $(DATA_FI_DIR)/Masks
	python scripts/infer_inpainting.py --frames-dir $(word 1,$^) \
                                       --masks-dir $(word 2,$^) \
                                       --results-dir $(dir $@) \
                                       --inpainting-model $(FLOW_INPAINTING_MODEL) \
                                       --flow-model 'None' \
                                       --mode 'flow_inpainting'

$(DATA_FI_DIR)/Flows $(DATA_FI_DIR)/Masks &: $(DATA_INTERIM_DIR)/$(FLOW_DATASET)/Flows \
											 $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/Annotations \
											 $(DATA_INTERIM_DIR)/$(VIDEO_DATASET)/ObjectStats
	python scripts/data/prepare_dataset.py --frames-dir $(word 1,$^) \
										   --annotations-dir $(word 2,$^) \
										   --object-stats-dir $(word 3,$^) \
										   --processed-dir $(dir $@) \
										   --limit-samples $(SAMPLES) \
										   --seed $(SEED) \
										   --frame-type 'flow' \
										   --mode 'cross'

## End-to-end
ee : $(RESULTS_EE_DIR)/Benchmark \
     $(RESULTS_EE_DIR)/Initialization \
     $(RESULTS_EE_DIR)/Images \
     $(RESULTS_EE_DIR)/Masks

$(RESULTS_EE_DIR)/Benchmark $(RESULTS_EE_DIR)/Initialization $(RESULTS_EE_DIR)/Images $(RESULTS_EE_DIR)/Masks &: $(DATA_EE_DIR)/Images \
                                                                                                                 $(DATA_EE_DIR)/Annotations
	python scripts/infer_end2end.py --images-dir $(word 1,$^) \
                                    --annotations-dir $(word 2,$^) \
						            --results-dir $(dir $@) \
                                    --crop $(CROP_RATIO) \
						            --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
						            --dilation-size $(DILATION_SIZE) \
						            --dilation-iterations $(DILATION_ITERATIONS) \
						            --flow-model $(FLOW_MODEL) \
						            --inpainting-model $(IMAGE_INPAINTING_MODEL)
