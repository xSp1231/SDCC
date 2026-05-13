#!/usr/bin/env bash
NET=$1
NUNMGPU=$2
EXPNAME=$3
SPLIT_ID=$4

# bash run_voc_fsod_finetuning.sh r101 4 scac 1

SAVEDIR=./checkpoints/${EXPNAME}
PRTRAINEDMODEL=./ImageNetPretrained

if [ "$NET"x = "r101"x ]; then
  IMAGENET_PRETRAIN=${PRTRAINEDMODEL}/MSRA/R-101.pkl
  IMAGENET_PRETRAIN_TORCH=${PRTRAINEDMODEL}/torchvision/resnet101-5d3b4d8f.pth
fi

# ------------------------------ CGCL 閰嶇疆 ----------------------------------- #
CGCL_ENABLE=True            # CGCL 鎬诲紑鍏? True / False
CGCL_CONTAINER=2            # 姣忕被鏈€澶氬瓨 shots脳container 涓壒寰?CGCL_FEATURE_DIM=128        # 瀵规瘮澶存姇褰辩淮搴?CGCL_TAU=0.2                # 娓╁害绯绘暟 (瓒婂皬瀵规瘮瓒婂皷閿?
CGCL_COEF=0.5               # CGCL 鎹熷け鏉冮噸 (鐩稿浜?loss_cls 鍜?loss_box_reg)
# --------------------------------------------------------------------------- #

# ----------------------------- HN (Hard Negative) 閰嶇疆 --------------------- #
HN_ENABLE=False            # HN 鎬诲紑鍏? True / False
HN_ALPHA=0.3               # HN loss 鏉冮噸
HN_TAU=0.4                 # 闅捐礋鏍锋湰闃堝€?(鍓嶆櫙姒傜巼 > 蟿 鐨勮儗鏅)
# --------------------------------------------------------------------------- #

# ----------------------------- TFE (Text-guided Feature Enhancement) ------- #
TFE_ENABLE=True            # TFE 鎬诲紑鍏? True / False (寰皟闃舵浣跨敤)
TFE_TEXT_FEATURES_PATH="$(cd "$(dirname "$0")" && pwd)/text_features/voc_ViT-B-32_text_features.pt"
TFE_HIDDEN_DIM=256         # 浜ゅ弶娉ㄦ剰鍔涢殣钘忕淮搴?TFE_NUM_HEADS=4            # 娉ㄦ剰鍔涘ご鏁?# --------------------------------------------------------------------------- #

# ----------------------------- 鏍″噯鏂瑰紡閰嶇疆 --------------------------------- #
# PCB 鍜?CASC 鍙悓鏃跺紑鍚€佸崟鐙紑鍚垨鍏ㄩ儴鍏抽棴
# 鍚屾椂寮€鍚椂鎺ㄧ悊闃舵鍏?PCB 鏍″噯鍐?CASC 鏍″噯
CALIBRATION_MODE="CASC"  # 鍙€? PCB / CASC / BOTH / NONE
CASC_ALPHA=0.70          # CASC 鏍″噯铻嶅悎鏉冮噸 (妫€娴嬪櫒鍒嗘暟鐨勬瘮閲?
CASC_UPPER=1.0           # 楂樹簬璇ュ垎鏁扮殑妫€娴嬫涓嶅弬涓?CASC 鏍″噯
CASC_LOWER=0.05          # 浣庝簬璇ュ垎鏁扮殑妫€娴嬫涓嶅弬涓?CASC 鏍″噯
# --------------------------------------------------------------------------- #

# ------------------------------- Base Pre-train ---------------------------------- #
# python3 main.py --num-gpus ${NUNMGPU} --config-file configs/voc/defrcn_det_${NET}_base${SPLIT_ID}.yaml  \
#     --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                            \
#            OUTPUT_DIR ${SAVEDIR}/defrcn_det_${NET}_base${SPLIT_ID}

# ----------------------------- Model Preparation --------------------------------- #
python3 tools/model_surgery.py --dataset voc --method randinit                                \
    --src-path ${SAVEDIR}/defrcn_det_${NET}_base${SPLIT_ID}/model_final.pth                   \
    --save-dir ${SAVEDIR}/defrcn_det_${NET}_base${SPLIT_ID}
BASE_WEIGHT=${SAVEDIR}/defrcn_det_${NET}_base${SPLIT_ID}/model_reset_surgery.pth

# ------------------------------ Novel Fine-tuning (FSOD) ------------------------------- #

TOTAL_TASKS=$((1 * 5))  # 1 seed * 5 shots (鎸夐渶淇敼)
CURRENT_TASK=0

echo "========================================================================"
echo "寮€濮?FSOD Few-Shot 寰皟璁粌 (scac + CGCL)"
echo "========================================================================"
echo "閰嶇疆淇℃伅:"
echo "  - 缃戠粶: ${NET}"
echo "  - GPU鏁伴噺: ${NUNMGPU}"
echo "  - Split ID: ${SPLIT_ID}"
echo "  - 鎬讳换鍔℃暟: ${TOTAL_TASKS}"
echo "  - 鍩虹鏉冮噸: ${BASE_WEIGHT}"
echo "  - CGCL 鍚敤: ${CGCL_ENABLE}"
echo "  - CGCL 鍙傛暟: TAU=${CGCL_TAU}, COEF=${CGCL_COEF}, DIM=${CGCL_FEATURE_DIM}, CONTAINER=${CGCL_CONTAINER}"
echo "  - HN 鍚敤: ${HN_ENABLE} (ALPHA=${HN_ALPHA}, TAU=${HN_TAU})"
echo "  - TFE 鍚敤: ${TFE_ENABLE} (HIDDEN=${TFE_HIDDEN_DIM}, HEADS=${TFE_NUM_HEADS})"
echo "  - 鏍″噯鏂瑰紡: ${CALIBRATION_MODE}"
echo "========================================================================"
echo ""

for seed in 0
do
    for shot in 1 2 3 5 10
    do
        CURRENT_TASK=$((CURRENT_TASK + 1))

        TRAIN_NOVEL_NAME=voc_2007_trainval_novel${SPLIT_ID}_${shot}shot_seed${seed}
        TEST_NOVEL_NAME=voc_2007_test_novel${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_fsod_${NET}_novelx_${shot}shot_seedx.yaml
        OUTPUT_DIR=${SAVEDIR}/defrcn_fsod_${NET}_novel${SPLIT_ID}/${shot}shot_seed${seed}

        echo "========================================================================"
        echo "寮€濮嬭缁?[${CURRENT_TASK}/${TOTAL_TASKS}]"
        echo "========================================================================"
        echo "  Shot: ${shot}-shot"
        echo "  Seed: ${seed}"
        echo "  璁粌闆? ${TRAIN_NOVEL_NAME}"
        echo "  娴嬭瘯闆? ${TEST_NOVEL_NAME}"
        echo "  閰嶇疆鏂囦欢: ${CONFIG_PATH}"
        echo "  杈撳嚭鐩綍: ${OUTPUT_DIR}"
        echo "  寮€濮嬫椂闂? $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------------------------------------"

        START_TIME=$(date +%s)

        CGCL_OPTS=""
        if [ "${CGCL_ENABLE}" = "True" ]; then
            CGCL_OPTS="CGCL.ENABLE True \
                   CGCL.SHOTS ${shot} \
                   CGCL.CONTAINER ${CGCL_CONTAINER} \
                   CGCL.FEATURE_DIM ${CGCL_FEATURE_DIM} \
                   CGCL.KNOWLEDGE_MATRIX $(cd "$(dirname "$0")" && pwd)/KnowledgeMatrix/vision_relationship/split${SPLIT_ID}/cluster_5_replace_gt.npy \
                   CGCL.TAU ${CGCL_TAU} \
                   CGCL.COEF ${CGCL_COEF}"
        else
            CGCL_OPTS="CGCL.ENABLE False"
        fi

        HN_OPTS="MODEL.ROI_HEADS.HN_ENABLE False"
        if [ "${HN_ENABLE}" = "True" ]; then
            HN_OPTS="MODEL.ROI_HEADS.HN_ENABLE True MODEL.ROI_HEADS.HN_ALPHA ${HN_ALPHA} MODEL.ROI_HEADS.HN_TAU ${HN_TAU}"
        fi

        TFE_OPTS="MODEL.ROI_HEADS.TFE_ENABLE False"
        if [ "${TFE_ENABLE}" = "True" ]; then
            TFE_OPTS="MODEL.ROI_HEADS.TFE_ENABLE True MODEL.ROI_HEADS.TFE_TEXT_FEATURES_PATH ${TFE_TEXT_FEATURES_PATH} MODEL.ROI_HEADS.TFE_HIDDEN_DIM ${TFE_HIDDEN_DIM} MODEL.ROI_HEADS.TFE_NUM_HEADS ${TFE_NUM_HEADS}"
        fi

        CALIB_OPTS="TEST.PCB_ENABLE False TEST.CASC_ENABLE False"
        if [ "${CALIBRATION_MODE}" = "PCB" ]; then
            CALIB_OPTS="TEST.PCB_ENABLE True TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} TEST.CASC_ENABLE False"
        elif [ "${CALIBRATION_MODE}" = "CASC" ]; then
            CALIB_OPTS="TEST.PCB_ENABLE False TEST.CASC_ENABLE True TEST.CASC_ALPHA ${CASC_ALPHA} TEST.CASC_UPPER ${CASC_UPPER} TEST.CASC_LOWER ${CASC_LOWER}"
        elif [ "${CALIBRATION_MODE}" = "BOTH" ]; then
            CALIB_OPTS="TEST.PCB_ENABLE True TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} TEST.CASC_ENABLE True TEST.CASC_ALPHA ${CASC_ALPHA} TEST.CASC_UPPER ${CASC_UPPER} TEST.CASC_LOWER ${CASC_LOWER}"
        fi

        python3 main.py --num-gpus ${NUNMGPU} --config-file ${CONFIG_PATH}                    \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT}                                                \
                   OUTPUT_DIR ${OUTPUT_DIR}                                                     \
                   MODEL.ROI_HEADS.NAME "Res5ROIHeadsCGCL"                        \
                   DATASETS.TRAIN "('"${TRAIN_NOVEL_NAME}"',)"                                 \
                   DATASETS.TEST  "('"${TEST_NOVEL_NAME}"',)"                                  \
                   MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE DC                                    \
                   ${CALIB_OPTS}                                                                \
                   ${CGCL_OPTS}                                                                  \
                   ${HN_OPTS}                                                                   \
                   ${TFE_OPTS}

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        MINUTES=$((ELAPSED / 60))
        SECONDS=$((ELAPSED % 60))

        echo "------------------------------------------------------------------------"
        echo "瀹屾垚璁粌 [${CURRENT_TASK}/${TOTAL_TASKS}]"
        echo "  Shot: ${shot}-shot | Seed: ${seed}"
        echo "  缁撴潫鏃堕棿: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  鑰楁椂: ${MINUTES}鍒?{SECONDS}绉?
        echo "  杩涘害: $(awk "BEGIN {printf \"%.1f\", ${CURRENT_TASK}/${TOTAL_TASKS}*100}")%"
        echo "========================================================================"
        echo ""
    done
done

echo ""
echo "========================================================================"
echo "鎵€鏈?FSOD 寰皟浠诲姟瀹屾垚锛?
echo "========================================================================"
echo "  鎬讳换鍔℃暟: ${TOTAL_TASKS}"
echo "  瀹屾垚鏃堕棿: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  缁撴灉鐩綍: ${SAVEDIR}/defrcn_fsod_${NET}_novel${SPLIT_ID}"
echo "========================================================================"
echo ""

# 姹囨€绘墍鏈夌粨鏋?# python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_fsod_${NET}_novel${SPLIT_ID} --shot-list 1 2 3 5 10
