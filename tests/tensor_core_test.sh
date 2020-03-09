nv-nsight-cu-cli --csv --nvtx -f -o profile_synthetic_O0 python tests/tensor_core_test_synthetic.py O0
nv-nsight-cu-cli --csv --nvtx -f -o profile_synthetic_O1 python tests/tensor_core_test_synthetic.py O1

nv-nsight-cu-cli --csv --nvtx -f -o profile_real_O0 python tests/tensor_core_test_real.py O0
nv-nsight-cu-cli --csv --nvtx -f -o profile_real_O1 python tests/tensor_core_test_real.py O1

nv-nsight-cu-cli --csv --nvtx -f -o profile_baseline_O0 python tests/tensor_core_test_baseline.py O0
nv-nsight-cu-cli --csv --nvtx -f -o profile_baseline_O1 python tests/tensor_core_test_baseline.py O1
