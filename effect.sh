#!/bin/bash

# nohup python create_pretraining_data.py  \
#  --input_file=./training_data/S1_train \
#  --output_file=./training_data/S1.tfrecord \
#  --vocab_file=./training_data/vocab_atlas_single1.txt \
#  --do_lower_case=True \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --masked_lm_prob=0.15  \
#  --random_seed=12345  \
#  --dupe_factor=5 &

# nohup python create_pretraining_data.py  \
#   --input_file=./training_data/S2_train \
#   --output_file=./training_data/S2.tfrecord \
#   --vocab_file=./training_data/vocab_atlas_single2.txt \
#   --do_lower_case=True \
#   --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --masked_lm_prob=0.15  \
#  --random_seed=12345  \
#  --dupe_factor=5 &


#  nohup python create_pretraining_data.py  \
#  --input_file=./training_data/S3_train \
#  --output_file=./training_data/S3.tfrecord \
#  --vocab_file=./training_data/vocab_atlas_single3.txt \
#  --do_lower_case=True \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --masked_lm_prob=0.15  \
#  --random_seed=12345  \
#  --dupe_factor=5 &

# nohup python create_pretraining_data.py  \
#  --input_file=./training_data/S4_train \
#  --output_file=./training_data/S4.tfrecord \
#  --vocab_file=./training_data/vocab_atlas_single4.txt \
#  --do_lower_case=True \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --masked_lm_prob=0.15  \
#  --random_seed=12345  \
#  --dupe_factor=5 &


# nohup python create_pretraining_data.py  \
#   --input_file=./training_data/train_atlas_12.txt \
#   --output_file=./training_data/M12.tfrecord \
#   --vocab_file=./training_data/vocab_atlas_12.txt \
#   --do_lower_case=True \
#   --max_seq_length=32  \
#   --max_predictions_per_seq=20  \
#   --masked_lm_prob=0.15  \
#   --random_seed=12345  \
#   --dupe_factor=5 &

#  nohup python create_pretraining_data.py  \
#  --input_file=./training_data/train_atlas_34.txt \
#  --output_file=./training_data/M34.tfrecord \
#   --vocab_file=./training_data/vocab_atlas_34.txt \
#   --do_lower_case=True \
#   --max_seq_length=32  \
#   --max_predictions_per_seq=20  \
#   --masked_lm_prob=0.15  \
#   --random_seed=12345  \
#   --dupe_factor=5 &

#    nohup python create_pretraining_data.py  \
#  --input_file=./training_data/train_atlas_56.txt \
#  --output_file=./training_data/M56.tfrecord \
#   --vocab_file=./training_data/vocab_atlas_56.txt \
#   --do_lower_case=True \
#   --max_seq_length=32  \
#   --max_predictions_per_seq=20  \
#   --masked_lm_prob=0.15  \
#   --random_seed=12345  \
#   --dupe_factor=5 &
# wait

#   nohup python -u run_pretraining.py  \
#   --input_file=./training_data/S1.tfrecord \
#   --output_dir=./models/S1 \
#   --do_train=True \
#   --do_eval=True  \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./uncased_L-6_H-128_A-2/bert_model.ckpt \
#   --train_batch_size=4  \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --num_train_steps=10000  \
#  --num_warmup_steps=10  \
#  --gpu=1  \
#  --learning_rate=2e-5 > ./logs/trainS1.log &

#   nohup python -u run_pretraining.py  \
#   --input_file=./training_data/S2.tfrecord \
#   --output_dir=./models/S2 \
#   --do_train=True \
#   --do_eval=True  \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./uncased_L-6_H-128_A-2/bert_model.ckpt \
#   --train_batch_size=4  \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --num_train_steps=10000  \
#  --num_warmup_steps=10  \
#  --gpu=2  \
#  --learning_rate=2e-5 > ./logs/trainS2.log &

#   nohup python -u run_pretraining.py  \
#   --input_file=./training_data/S3.tfrecord \
#   --output_dir=./models/S3 \
#   --do_train=True \
#   --do_eval=True  \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./uncased_L-6_H-128_A-2/bert_model.ckpt \
#   --train_batch_size=4  \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --num_train_steps=10000  \
#  --num_warmup_steps=10  \
#  --gpu=3  \
#  --learning_rate=2e-5 > ./logs/trainS3.log &

#   nohup python -u run_pretraining.py  \
#   --input_file=./training_data/S4.tfrecord \
#   --output_dir=./models/S4 \
#   --do_train=True \
#   --do_eval=True  \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./uncased_L-6_H-128_A-2/bert_model.ckpt \
#   --train_batch_size=4  \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --num_train_steps=10000  \
#  --num_warmup_steps=10  \
#  --gpu=4  \
#  --learning_rate=2e-5 > ./logs/trainS4.log &

# wait

# nohup python -u run_pretraining.py  \
#   --input_file=./training_data/M12.tfrecord \
#   --output_dir=./models/M12  \
#   --do_train=True \
#   --do_eval=True  \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./uncased_L-6_H-128_A-2/bert_model.ckpt \
#   --train_batch_size=4  \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --num_train_steps=10000  \
#   --gpu=1  \
#  --num_warmup_steps=10  \
#  --learning_rate=2e-5 > ./logs/trainM12.log &

# nohup python -u run_pretraining.py  \
#   --input_file=./training_data/M34.tfrecord \
#   --output_dir=./models/M34  \
#   --do_train=True \
#   --do_eval=True  \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./uncased_L-6_H-128_A-2/bert_model.ckpt \
#   --train_batch_size=4  \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --num_train_steps=10000  \
#   --gpu=2  \
#  --num_warmup_steps=10  \
#  --learning_rate=2e-5 > ./logs/trainM34.log &

#  nohup python -u run_pretraining.py  \
#   --input_file=./training_data/M56.tfrecord \
#   --output_dir=./models/M56  \
#   --do_train=True \
#   --do_eval=True  \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./uncased_L-6_H-128_A-2/bert_model.ckpt \
#   --train_batch_size=4  \
#  --max_seq_length=32  \
#  --max_predictions_per_seq=20  \
#  --num_train_steps=10000  \
#   --gpu=3  \
#  --num_warmup_steps=10  \
#  --learning_rate=2e-5 > ./logs/trainM56.log &

# wait

# extract embeddings

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/S1_benign \
#   --output_file=./embedding_data/S1_benign.json  \
#   --vocab_file=./training_data/vocab_atlas_single1.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/S1/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=1  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_S1.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/S2_benign \
#   --output_file=./embedding_data/S2_benign.json  \
#   --vocab_file=./training_data/vocab_atlas_single2.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/S2/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=2  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_S2.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/S3_benign \
#   --output_file=./embedding_data/S3_benign.json  \
#   --vocab_file=./training_data/vocab_atlas_single3.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/S3/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=3  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_S3.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/S4_benign \
#   --output_file=./embedding_data/S4_benign.json  \
#   --vocab_file=./training_data/vocab_atlas_single4.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/S4/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=4  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_S4.log &

# wait

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/S1_test \
#   --output_file=./embedding_data/S1_test.json  \
#   --vocab_file=./training_data/vocab_atlas_single1.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/S1/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=1  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_S1_test.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/S2_test \
#   --output_file=./embedding_data/S2_test.json  \
#   --vocab_file=./training_data/vocab_atlas_single2.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#     --init_checkpoint=./models/S2/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=2  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_S2_test.log &


#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/S3_test \
#   --output_file=./embedding_data/S3_test.json  \
#   --vocab_file=./training_data/vocab_atlas_single3.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#    --init_checkpoint=./models/S3/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=3  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_S3_test.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/S4_test \
#   --output_file=./embedding_data/S4_test.json  \
#   --vocab_file=./training_data/vocab_atlas_single4.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/S4/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=4  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_S4_test.log &
# wait


# nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M12_benign \
#   --output_file=./embedding_data/M12_benign.json  \
#   --vocab_file=./training_data/vocab_atlas_12.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M12/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=1  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M12.log &


# nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M34_benign \
#   --output_file=./embedding_data/M34_benign.json  \
#   --vocab_file=./training_data/vocab_atlas_34.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M34/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=2  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M34.log &


#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M56_benign \
#   --output_file=./embedding_data/M56_benign.json  \
#   --vocab_file=./training_data/vocab_atlas_56.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M56/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=3  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M56.log &

# wait

# nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M121_test \
#   --output_file=./embedding_data/M121_test.json  \
#   --vocab_file=./training_data/vocab_atlas_12.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M12/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=1  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M121_test.log &


# nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M122_test \
#   --output_file=./embedding_data/M122_test.json  \
#   --vocab_file=./training_data/vocab_atlas_12.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M12/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=2  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M122_test.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M123_test \
#   --output_file=./embedding_data/M123_test.json  \
#   --vocab_file=./training_data/vocab_atlas_12.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M12/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=3  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M123_test.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M124_test \
#   --output_file=./embedding_data/M124_test.json  \
#   --vocab_file=./training_data/vocab_atlas_12.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M12/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=4  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M124_test.log &

# wait

# nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M341_test \
#   --output_file=./embedding_data/M341_test.json  \
#   --vocab_file=./training_data/vocab_atlas_34.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M34/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=1  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M341_test.log &


# nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M342_test \
#   --output_file=./embedding_data/M342_test.json  \
#   --vocab_file=./training_data/vocab_atlas_34.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M34/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=2  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M342_test.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M343_test \
#   --output_file=./embedding_data/M343_test.json  \
#   --vocab_file=./training_data/vocab_atlas_34.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M34/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=3  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M343_test.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M344_test \
#   --output_file=./embedding_data/M344_test.json  \
#   --vocab_file=./training_data/vocab_atlas_34.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M34/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=4  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M344_test.log &

#   wait


#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M561_test \
#   --output_file=./embedding_data/M561_test.json  \
#   --vocab_file=./training_data/vocab_atlas_56.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M56/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=1  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M561_test.log &


# nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M562_test \
#   --output_file=./embedding_data/M562_test.json  \
#   --vocab_file=./training_data/vocab_atlas_56.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M56/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=2  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M562_test.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M563_test \
#   --output_file=./embedding_data/M563_test.json  \
#   --vocab_file=./training_data/vocab_atlas_56.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M56/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=3  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M563_test.log &

#   nohup python -u extract_multi_process2.py \
#   --input_file=./training_data/M564_test \
#   --output_file=./embedding_data/M564_test.json  \
#   --vocab_file=./training_data/vocab_atlas_56.txt \
#   --bert_config_file=./uncased_L-6_H-128_A-2/bert_config.json \
#   --init_checkpoint=./models/M56/model.ckpt-10000 \
#   --layers=-1  \
#   --gpu=4  \
#   --max_seq_length=32 \
#   --batch_size=2048 > ./logs/extract_M564_test.log &

  # wait

nohup python -u evaluate_onesvm_Sdatasets.py -flag 1 -nu 0.1 -gama 0.1 -gpu 2 > logs/S1.log &
nohup python -u evaluate_onesvm_Sdatasets.py -flag 2 -nu 0.1 -gama 0.15 -gpu 3 > logs/S2.log &
nohup python -u evaluate_onesvm_Sdatasets.py -flag 3 -nu 0.1 -gama 0.2 -gpu 4 > logs/S3.log &
nohup python -u evaluate_onesvm_Sdatasets.py -flag 4 -nu 0.1 -gama 0.15 -gpu 5 > logs/S4.log &
wait

nohup python -u evaluate_onesvm_12_ground_modify.py -nu 0.08 -gama 0.3 -gpu 2 > logs/M12.log &
nohup python -u evaluate_onesvm_34_ground_modify.py -nu 0.08 -gama 0.3 -gpu 3 > logs/M34.log &
nohup python -u evaluate_onesvm_56_ground_modify.py -nu 0.08 -gama 0.3 -gpu 4 > logs/M56.log &
wait
