python fine_tuning.py \
    --model-saving-dir /home/alien/XUEYu/paper_code/Parameters/2018_CPC \
    --logging-dir      /home/alien/XUEYu/paper_code/Parameters/Log \
    --timestep         12   \
    --state-archi      True \
    --hidden-GRU       5 \
    --state-parallel   False \
    --index            9 \
    --load-pretraining-params True \
    --timestep-state   True \
    --classifier       class_1 \
    --gru-layers       1 \
    --encoder-mode     False \

<< 'COMMENT'
python main.py \
    --model-saving-dir /home/alien/XUEYu/paper_code/Parameters/2018_CPC \
    --logging-dir      /home/alien/XUEYu/paper_code/Parameters/Log  \
    --epochs           200 \
    --batch-size       64 \
    --timestep         12 \
    --state-archi      True \
    --hidden-GRU       5 \
    --state-parallel   False \
    --gru-layers       1 \
COMMENT