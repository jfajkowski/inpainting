.PHONY: all install data flow_estimation flow_inpainting image_inpainting tracking_and_segmentation
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

DATA_DIR = data
DATA_RAW_DIR = $(DATA_DIR)/raw
DATA_INTERIM_DIR = $(DATA_DIR)/interim
DATA_PROCESSED_DIR = $(DATA_DIR)/processed
DATA_FLOW_ESTIMATION_DIR = $(DATA_PROCESSED_DIR)/flow_estimation
DATA_FLOW_INPAINTING_DIR = $(DATA_PROCESSED_DIR)/flow_inpainting
DATA_IMAGE_INPAINTING_DIR = $(DATA_PROCESSED_DIR)/image_inpainting
DATA_TRACKING_AND_SEGMENTATION_DIR = $(DATA_PROCESSED_DIR)/tracking_and_segmentation

RESULTS_DIR = results
RESULTS_FLOW_ESTIMATION_DIR = $(RESULTS_DIR)/flow_estimation/$(basename $(CONFIG))
RESULTS_FLOW_INPAINTING_DIR = $(RESULTS_DIR)/flow_inpainting/$(basename $(CONFIG))
RESULTS_IMAGE_INPAINTING_DIR = $(RESULTS_DIR)/image_inpainting/$(basename $(CONFIG))
RESULTS_TRACKING_AND_SEGMENTATION_DIR = $(RESULTS_DIR)/tracking_and_segmentation/$(basename $(CONFIG))

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
install :
	conda env create -f environment.yml
	cd ./inpainting/external/layers/correlation_package && python setup.py install
	cd ./inpainting/external/layers/resample2d_package && python setup.py install
	cd ./inpainting/external/layers/channelnorm_package && python setup.py install

all : data flow_estimation flow_inpainting image_inpainting tracking_and_segmentation videos

#################################################################################
# DATA RULES                                                                    #
#################################################################################

data : $(DATA_INTERIM_DIR)/DAVIS/Annotations/480p \
       $(DATA_INTERIM_DIR)/DAVIS/JPEGImages/480p \
       $(DATA_INTERIM_DIR)/DAVIS/ObjectStats \
       $(DATA_INTERIM_DIR)/MPI-Sintel-complete/training/final \
       $(DATA_INTERIM_DIR)/MPI-Sintel-complete/training/flow

$(DATA_INTERIM_DIR)/DAVIS/Annotations/480p : $(DATA_RAW_DIR)/DAVIS/Annotations/480p
	python scripts/data/adjust_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --crop $(CROP_WIDTH) $(CROP_HEIGHT) \
                                         --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
                                         --frame-type 'annotation'

$(DATA_INTERIM_DIR)/DAVIS/JPEGImages/480p : $(DATA_RAW_DIR)/DAVIS/JPEGImages/480p
	python scripts/data/adjust_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --crop $(CROP_WIDTH) $(CROP_HEIGHT) \
                                         --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
                                         --frame-type 'image'

$(DATA_INTERIM_DIR)/DAVIS/ObjectStats : $(DATA_INTERIM_DIR)/DAVIS/Annotations/480p
	python scripts/data/calculate_object_stats.py --annotations-dir $(word 1,$^) \
                                                  --object-stats-dir $@ \
												  --min-presence $(MIN_PRESENCE) \
												  --min-mean-size $(MIN_MEAN_SIZE) \
												  --max-mean-size $(MAX_MEAN_SIZE)

$(DATA_INTERIM_DIR)/MPI-Sintel-complete/training/final : $(DATA_RAW_DIR)/MPI-Sintel-complete/training/final
	python scripts/data/adjust_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --crop $(CROP_WIDTH) $(CROP_HEIGHT) \
                                         --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
                                         --frame-type 'image'

$(DATA_INTERIM_DIR)/MPI-Sintel-complete/training/flow : $(DATA_RAW_DIR)/MPI-Sintel-complete/training/flow
	python scripts/data/adjust_frames.py --frames-dir $(word 1,$^) \
                                         --interim-dir $@ \
                                         --crop $(CROP_WIDTH) $(CROP_HEIGHT) \
                                         --scale $(SCALE_WIDTH) $(SCALE_HEIGHT) \
                                         --frame-type 'flow'

#################################################################################
# EXPERIMENTS                                                                   #
#################################################################################


## Flow estimation
flow_estimation : $(RESULTS_FLOW_ESTIMATION_DIR)/Benchmark \
                  $(RESULTS_FLOW_ESTIMATION_DIR)/Evaluation \
                  $(RESULTS_FLOW_ESTIMATION_DIR)/Flows

$(RESULTS_FLOW_ESTIMATION_DIR)/Evaluation : $(RESULTS_FLOW_ESTIMATION_DIR)/Flows \
                                            $(DATA_FLOW_ESTIMATION_DIR)/Flows
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode 'flow_estimation'

$(RESULTS_FLOW_ESTIMATION_DIR)/Benchmark $(RESULTS_FLOW_ESTIMATION_DIR)/Flows &: $(DATA_FLOW_ESTIMATION_DIR)/Images
	python scripts/infer_flow_estimation.py --images-dir $(word 1,$^) \
                                            --results-dir $(dir $@) \
                                            --flow-model $(FLOW_MODEL)

$(DATA_FLOW_ESTIMATION_DIR)/Images $(DATA_FLOW_ESTIMATION_DIR)/Flows &: $(DATA_INTERIM_DIR)/MPI-Sintel-complete/training/final \
                                                                        $(DATA_INTERIM_DIR)/MPI-Sintel-complete/training/flow
	python scripts/data/prepare_flow_estimation_dataset.py --images-dir $(word 1,$^) \
										   				   --flows-dir $(word 2,$^) \
													       --processed-dir $(dir $@) \
													       --limit-samples $(SAMPLES) \
													       --seed $(SEED)


## Flow estimation
flow_inpainting : $(RESULTS_FLOW_INPAINTING_DIR)/Benchmark \
				  $(RESULTS_FLOW_INPAINTING_DIR)/Evaluation \
                  $(RESULTS_FLOW_INPAINTING_DIR)/Flows

$(RESULTS_FLOW_INPAINTING_DIR)/Evaluation : $(RESULTS_FLOW_INPAINTING_DIR)/Flows \
                                            $(DATA_FLOW_INPAINTING_DIR)/Flows
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode 'flow_inpainting'

$(RESULTS_FLOW_INPAINTING_DIR)/Benchmark $(RESULTS_FLOW_INPAINTING_DIR)/Flows &: $(DATA_FLOW_INPAINTING_DIR)/Flows \
                                                                                 $(DATA_FLOW_INPAINTING_DIR)/Masks
	python scripts/infer_inpainting.py --frames-dir $(word 1,$^) \
                                       --masks-dir $(word 2,$^) \
                                       --results-dir $(dir $@) \
                                       --inpainting-model $(FLOW_INPAINTING_MODEL) \
                                       --mode 'flow_inpainting'

$(DATA_FLOW_INPAINTING_DIR)/Flows $(DATA_FLOW_INPAINTING_DIR)/Masks &: $(DATA_INTERIM_DIR)/MPI-Sintel-complete/training/flow \
                                                                       $(DATA_INTERIM_DIR)/DAVIS/Annotations/480p \
                                                                       $(DATA_INTERIM_DIR)/DAVIS/ObjectStats
	python scripts/data/prepare_dataset.py --frames-dir $(word 1,$^) \
										   --annotations-dir $(word 2,$^) \
										   --object-stats-dir $(word 3,$^) \
										   --processed-dir $(dir $@) \
										   --limit-samples $(SAMPLES) \
										   --seed $(SEED) \
										   --mode 'flow_inpainting'


## Image inpainting
image_inpainting : $(RESULTS_IMAGE_INPAINTING_DIR)/Benchmark \
				   $(RESULTS_IMAGE_INPAINTING_DIR)/Evaluation \
                   $(RESULTS_IMAGE_INPAINTING_DIR)/Images

$(RESULTS_IMAGE_INPAINTING_DIR)/Evaluation : $(RESULTS_IMAGE_INPAINTING_DIR)/Images \
                                             $(DATA_IMAGE_INPAINTING_DIR)/Images
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode 'image_inpainting'

$(RESULTS_IMAGE_INPAINTING_DIR)/Benchmark $(RESULTS_IMAGE_INPAINTING_DIR)/Images &: $(DATA_IMAGE_INPAINTING_DIR)/Images \
                                                                                    $(DATA_IMAGE_INPAINTING_DIR)/Masks
	python scripts/infer_inpainting.py --frames-dir $(word 1,$^) \
                                       --masks-dir $(word 2,$^) \
                                       --results-dir $(dir $@) \
                                       --inpainting-model $(IMAGE_INPAINTING_MODEL) \
                                       --mode 'image_inpainting'

$(DATA_IMAGE_INPAINTING_DIR)/Images $(DATA_IMAGE_INPAINTING_DIR)/Masks &: $(DATA_INTERIM_DIR)/DAVIS/JPEGImages/480p \
                                                                          $(DATA_INTERIM_DIR)/DAVIS/Annotations/480p \
                                                                          $(DATA_INTERIM_DIR)/DAVIS/ObjectStats
	python scripts/data/prepare_dataset.py --frames-dir $(word 1,$^) \
										   --annotations-dir $(word 2,$^) \
										   --object-stats-dir $(word 3,$^) \
										   --processed-dir $(dir $@) \
										   --limit-samples $(SAMPLES) \
										   --seed $(SEED) \
										   --mode 'image_inpainting'


## Tracking and segmentation
tracking_and_segmentation : $(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/Benchmark \
							$(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/Evaluation \
						    $(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/Initialization \
						    $(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/Masks

$(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/Evaluation : $(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/Masks \
                                                      $(DATA_TRACKING_AND_SEGMENTATION_DIR)/Masks
	python scripts/evaluate.py --output-frames-dir $(word 1,$^) \
                               --target-frames-dir $(word 2,$^) \
                               --results-dir $(dir $@) \
                               --mode 'tracking_and_segmentation'

$(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/Benchmark $(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/Initialization $(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/Masks &: $(DATA_TRACKING_AND_SEGMENTATION_DIR)/Images \
                                                                                                                                                             $(DATA_TRACKING_AND_SEGMENTATION_DIR)/Masks
	python scripts/infer_tracking_and_segmentation.py --images-dir $(word 1,$^) \
													  --masks-dir $(word 2,$^) \
													  --results-dir $(dir $@) \
													  --dilation-size $(DILATION_SIZE) \
													  --dilation-iterations $(DILATION_ITERATIONS)


$(DATA_TRACKING_AND_SEGMENTATION_DIR)/Images $(DATA_TRACKING_AND_SEGMENTATION_DIR)/Masks &: $(DATA_INTERIM_DIR)/DAVIS/JPEGImages/480p \
																						    $(DATA_INTERIM_DIR)/DAVIS/Annotations/480p \
																						    $(DATA_INTERIM_DIR)/DAVIS/ObjectStats
	python scripts/data/prepare_dataset.py --frames-dir $(word 1,$^) \
										   --annotations-dir $(word 2,$^) \
										   --object-stats-dir $(word 3,$^) \
										   --processed-dir $(dir $@) \
										   --limit-samples $(SAMPLES) \
										   --seed $(SEED) \
										   --mode 'tracking_and_segmentation'


## Videos
videos : $(DATA_FLOW_ESTIMATION_DIR)/ImageVideos $(DATA_FLOW_ESTIMATION_DIR)/FlowVideos $(RESULTS_FLOW_ESTIMATION_DIR)/FlowVideos \
         $(DATA_FLOW_INPAINTING_DIR)/FlowVideos $(DATA_FLOW_INPAINTING_DIR)/MaskVideos $(RESULTS_FLOW_INPAINTING_DIR)/FlowVideos \
         $(DATA_IMAGE_INPAINTING_DIR)/ImageVideos $(DATA_IMAGE_INPAINTING_DIR)/MaskVideos $(RESULTS_IMAGE_INPAINTING_DIR)/ImageVideos \
         $(DATA_TRACKING_AND_SEGMENTATION_DIR)/ImageVideos $(DATA_TRACKING_AND_SEGMENTATION_DIR)/MaskVideos $(RESULTS_TRACKING_AND_SEGMENTATION_DIR)/MaskVideos

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
