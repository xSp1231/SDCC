export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
export PATH=$PATH:/usr/local/cuda-10.1/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.1

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

for SPLIT_ID in 1
do
    for shot in 1   # if final, 10 -> 1 2 3 5 10
    do
        for seed in 0
        do
            for repeat in 1 2 3 4 5 6
            do
                CUDA_VISIBLE_DEVICES=2,3,4,5 python main.py --num-gpus 4 --config-file configs/voc/defrcn_fsod_r101_all${SPLIT_ID}_${shot}shot_seed${seed}.yaml --opts CGCL.TAU 0.2 CGCL.COEF 0.5 TEST.PCB_ENABLE True OUTPUT_DIR checkpoints/voc/split${SPLIT_ID}/${shot}shot_CGCL_seed${seed}_w_PCB_${repeat}
            done
        done
    done
done
