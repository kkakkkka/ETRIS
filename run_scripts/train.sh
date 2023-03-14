dataset_name="refcoco" # "refcoco", "refcoco+", "refcocog_g", "refcocog_u"
config_name="bridge_r101.yaml"
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python3 -u train.py \
      --config config/$dataset_name/$config_name