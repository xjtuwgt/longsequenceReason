#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=multihopQA
SAVE_PATH=model
DATA_PATH=data/hotpotqa/distractor_qa

#The first four parameters must be provided
PAIR_SCORE=$1
TRIPLE_SCORE=$2
SAVE_ID=$3
SAVE=$SAVE_PATH/"$SAVE_ID"

#Only used in training
PROJECT_DIM=$4
BATCH_SIZE=$5
MAX_DOC_NUM=$6
LEARNING_RATE=$7
FEA_DROP=$8
ATT_DROP=$9
WEIGHTDECAY=${10}
EPOCH=${11}
SEED=${12}
SENT_THRETH=${13}
MASK_NAME=${14}
FROZEN=${15}
TRAIN_DA_TYPE=${16}
TRAIN_SHUFFLE=${17}
SPAN_WEIGHT=${18}
PAIR_SCORE_WEIGHT=${19}
WITH_GRAPH=${20}
WITH_GRAPH_TRAIN=${21}
TASK_NAME=${22}
PRE_TRAIN=${23}

echo "Start Training......"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u $CODE_PATH/qarun.py --do_train \
    --cuda \
    --score_model_name $PAIR_SCORE\
    --hop_model_name $TRIPLE_SCORE\
    --data_path $DATA_PATH\
    --epoch $EPOCH\
    --save_path $SAVE\
    --input_drop $FEA_DROP\
    --batch_size $BATCH_SIZE\
    --max_doc_num $MAX_DOC_NUM\
    --attn_drop $ATT_DROP\
    --project_dim $PROJECT_DIM\
    --learning_rate $LEARNING_RATE\
    --weight_decay $WEIGHTDECAY\
    --rand_seed $SEED\
    --sent_threshold $SENT_THRETH\
    --global_mask_type $MASK_NAME\
    --frozen_layer_num $FROZEN\
    --train_data_filtered $TRAIN_DA_TYPE\
    --training_shuffle $TRAIN_SHUFFLE\
    --span_weight $SPAN_WEIGHT\
    --pair_score_weight $PAIR_SCORE_WEIGHT\
    --with_graph $WITH_GRAPH\
    --with_graph_training $WITH_GRAPH_TRAIN\
    --task $TASK_NAME\
    --pretrained_cfg_flag $PRE_TRAIN