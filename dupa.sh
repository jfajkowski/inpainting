task() {
    RESULT_DIR=$1
    cat ${RESULT_DIR}/End2End/mean.txt ${RESULT_DIR}/Inpainter/mean.txt ${RESULT_DIR}/Tracker/mean.txt > ${RESULT_DIR}/results.txt
#    python scripts/evaluate_inpainter.py --output-images-dir ${RESULT_DIR}/End2End/OutputImages --target-images-dir data/processed/$(basename ${RESULT_DIR})/TargetImages --results-dir ${RESULT_DIR}/End2End &
#    python scripts/evaluate_inpainter.py --output-images-dir ${RESULT_DIR}/Inpainter/OutputImages --target-images-dir data/processed/$(basename ${RESULT_DIR})/TargetImages --results-dir ${RESULT_DIR}/Inpainter &
#    python scripts/evaluate_tracker.py --output-masks-dir ${RESULT_DIR}/Tracker/OutputMasks --target-masks-dir data/processed/$(basename ${RESULT_DIR})/Masks --results-dir ${RESULT_DIR}/Tracker &
}

for RESULT_DIR in results/*/*; do task ${RESULT_DIR}; done