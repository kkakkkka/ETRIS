dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="bridge_r101.yaml"
gpu=0
split_name="testA" # "val", "testA", "testB" 
# Evaluation on the specified of the specified dataset
CUDA_VISIBLE_DEVICES=$gpu python3 -u test.py \
      --config config/$dataset_name/$config_name \
      --opts TEST.test_split $split_name \
             TEST.test_lmdb datasets/lmdb/$dataset_name/$split_name.lmdb
