python main.py \
    --model-saving-dir /home/alien/XUEYu/paper_code/Parameters/2018_Moco \
    --logging-dir      /home/alien/XUEYu/paper_code/Parameters/Log_Moco  \
    --epochs           200 \
    --batch-size       64 \
    --state-archi      False \
    --continue-train   True \


<< 'COMMENT'
python fine_tuning.py \
    --model-saving-dir /home/alien/XUEYu/paper_code/Parameters/2018_Moco \
    --logging-dir      /home/alien/XUEYu/paper_code/Parameters/Log_Moco \
    --state-archi      True \
    --index            1 \
    --load-pretraining-params False \
# pre-training model index.
COMMENT