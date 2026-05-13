#!/usr/bin/env bash

# ============================================================================
# CASC 校准评估脚本（仅推理，不训练）
# 用法: bash run_CASC_eval.sh r101 1 scac 1
# ============================================================================

NET=$1
NUNMGPU=$2
EXPNAME=$3
SPLIT_ID=$4

# ----------------------------- 路径配置 ------------------------------------- #
SAVEDIR=./checkpoint/${EXPNAME}

# 已训练好的模型所在子目录（shot 为变量，会在循环中替换）
MODEL_SUBDIR=defrcn_gfsod_${NET}_novel${SPLIT_ID}/defrcn_ccl_clip/exp3

# ----------------------------- CASC 配置 ---------------------------------- #
CASC_ALPHA=0.70
CASC_UPPER=1.0
CASC_LOWER=0.05
# ---------------------------------------------------------------------------- #

TOTAL_TASKS=5
CURRENT_TASK=0

echo "========================================================================"
echo " CASC plus校准评估（仅推理）"
echo "========================================================================"
echo "配置信息:"
echo "  - 网络: ${NET}"
echo "  - GPU数量: ${NUNMGPU}"
echo "  - Split ID: ${SPLIT_ID}"
echo "  - 模型目录: ${SAVEDIR}/${MODEL_SUBDIR}"
echo "  - CASC_ALPHA: ${CASC_ALPHA}"
echo "  - CASC_UPPER: ${CASC_UPPER}"
echo "  - CASC_LOWER: ${CASC_LOWER}"
echo "========================================================================"
echo ""

for seed in 0
do
    for shot in 1 2 3 5 10
    do
        CURRENT_TASK=$((CURRENT_TASK + 1))

        TRAIN_ALL_NAME=voc_2007_trainval_all${SPLIT_ID}_${shot}shot_seed${seed}
        TEST_ALL_NAME=voc_2007_test_all${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_gfsod_${NET}_novelx_${shot}shot_seedx.yaml

        MODEL_WEIGHTS=${SAVEDIR}/${MODEL_SUBDIR}/${shot}shot_seed${seed}/model_final.pth
        OUTPUT_DIR=${SAVEDIR}/${MODEL_SUBDIR}/${shot}shot_seed${seed}/CASC_eval

        echo "========================================================================"
        echo "开始评估 [${CURRENT_TASK}/${TOTAL_TASKS}]"
        echo "  Shot: ${shot}-shot | Seed: ${seed}"
        echo "  模型权重: ${MODEL_WEIGHTS}"
        echo "  输出目录: ${OUTPUT_DIR}"
        echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------------------------------------"

        START_TIME=$(date +%s)

        python3 main.py --num-gpus ${NUNMGPU} --eval-only                                  \
            --config-file ${CONFIG_PATH}                                                    \
            --opts MODEL.WEIGHTS ${MODEL_WEIGHTS}                                           \
                   OUTPUT_DIR ${OUTPUT_DIR}                                                  \
                   MODEL.ROI_HEADS.NAME "Res5ROIHeadsCGCL"                     \
                   DATASETS.TRAIN "('"${TRAIN_ALL_NAME}"',)"                                \
                   DATASETS.TEST  "('"${TEST_ALL_NAME}"',)"                                 \
                   TEST.PCB_ENABLE False                                                     \
                   TEST.CASC_ENABLE True                                                   \
                   TEST.CASC_ALPHA ${CASC_ALPHA}                                        \
                   TEST.CASC_UPPER ${CASC_UPPER}                                        \
                   TEST.CASC_LOWER ${CASC_LOWER}

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        MINUTES=$((ELAPSED / 60))
        SECONDS=$((ELAPSED % 60))

        echo "------------------------------------------------------------------------"
        echo "完成评估 [${CURRENT_TASK}/${TOTAL_TASKS}]"
        echo "  Shot: ${shot}-shot | Seed: ${seed}"
        echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  耗时: ${MINUTES}分${SECONDS}秒"
        echo "========================================================================"
        echo ""
    done
done

echo ""
echo "========================================================================"
echo "所有 CASC 校准评估任务完成！"
echo "  完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果目录: ${SAVEDIR}/${MODEL_SUBDIR}/*/CASC_eval"
echo "========================================================================"
