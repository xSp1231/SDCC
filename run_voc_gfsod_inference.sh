#!/usr/bin/env bash
# =============================================================
# VOC GFSOD 纯推理脚本（仅 eval，不训练）
# 用法: bash run_voc_gfsod_inference.sh r101 1 scac 1
# 参数:
#   $1 NET       网络类型, 如 r101
#   $2 NUMGPU    GPU 数量
#   $3 EXPNAME   实验名称, 对应 checkpoint 根目录
#   $4 SPLIT_ID  VOC split, 1/2/3
# =============================================================
NET=$1
NUMGPU=$2
EXPNAME=$3
SPLIT_ID=$4

SAVEDIR=./checkpoint/${EXPNAME}

# ===================== 要评测的实验范围 ===================== #
EXP_TIMES=1          # 对应训练时的 EXP_TIMES
# ============================================================ #

# ========================= 校准配置 ========================= #
CALIBRATION_MODE="CASC"   # PCB / CASC / BOTH / NONE
CASC_ALPHA=0.70
CASC_UPPER=1.0
CASC_LOWER=0.05
# ============================================================ #

echo "========================================================================"
echo " VOC GFSOD 纯推理评测"
echo "  NET=${NET} | GPU=${NUMGPU} | SPLIT_ID=${SPLIT_ID}"
echo "  校准方式: ${CALIBRATION_MODE}"
echo "========================================================================"

for exp_idx in $(seq 1 ${EXP_TIMES})
do
    echo "===== 第 ${exp_idx}/${EXP_TIMES} 次实验 ====="

    for seed in 0
    do
        for shot in 1
        do
            CONFIG_PATH=configs/voc/defrcn_gfsod_${NET}_novelx_${shot}shot_seedx.yaml
            OUTPUT_DIR=${SAVEDIR}/defrcn_gfsod_${NET}_novel${SPLIT_ID}/p1_defrcn_ccl_tfe_clip/re_exp3/${shot}shot_seed${seed}
            TEST_ALL_NAME=voc_2007_test_all${SPLIT_ID}
            TRAIN_ALL_NAME=voc_2007_trainval_all${SPLIT_ID}_${shot}shot_seed${seed}
            MODEL_WEIGHT=${OUTPUT_DIR}/model_final.pth

            if [ ! -f "${MODEL_WEIGHT}" ]; then
                echo "[跳过] 权重不存在: ${MODEL_WEIGHT}"
                continue
            fi

            echo "------------------------------------------------------------------------"
            echo "推理: exp${exp_idx} | ${shot}-shot | seed${seed}"
            echo "  权重: ${MODEL_WEIGHT}"
            echo "  测试集: ${TEST_ALL_NAME}"
            echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "------------------------------------------------------------------------"

            # ----------- 校准配置 ----------- #
            CALIB_OPTS="TEST.PCB_ENABLE False TEST.CASC_ENABLE False"
            if [ "${CALIBRATION_MODE}" = "PCB" ]; then
                CALIB_OPTS="TEST.PCB_ENABLE True TEST.CASC_ENABLE False"
            elif [ "${CALIBRATION_MODE}" = "CASC" ]; then
                CALIB_OPTS="TEST.PCB_ENABLE False \
                            TEST.CASC_ENABLE True \
                            TEST.CASC_ALPHA ${CASC_ALPHA} \
                            TEST.CASC_UPPER ${CASC_UPPER} \
                            TEST.CASC_LOWER ${CASC_LOWER}"
            elif [ "${CALIBRATION_MODE}" = "BOTH" ]; then
                CALIB_OPTS="TEST.PCB_ENABLE True \
                            TEST.CASC_ENABLE True \
                            TEST.CASC_ALPHA ${CASC_ALPHA} \
                            TEST.CASC_UPPER ${CASC_UPPER} \
                            TEST.CASC_LOWER ${CASC_LOWER}"
            fi

            python3 main.py --num-gpus ${NUMGPU}    \
                --config-file ${CONFIG_PATH}         \
                --eval-only                          \
                --opts                               \
                    MODEL.WEIGHTS ${MODEL_WEIGHT}    \
                    OUTPUT_DIR ${OUTPUT_DIR}         \
                    MODEL.ROI_HEADS.NAME "Res5ROIHeadsCGCL" \
                    DATASETS.TRAIN "('"${TRAIN_ALL_NAME}"',)"  \
                    DATASETS.TEST  "('"${TEST_ALL_NAME}"',)"   \
                    MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE DC   \
                    ${CALIB_OPTS}

            echo "  完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "========================================================================"
        done
    done
done

echo ""
echo "========================================================================"
echo "所有推理任务完成！"
echo "========================================================================"
