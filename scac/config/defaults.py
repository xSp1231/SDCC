from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

_CC = _C

# ----------- Backbone ----------- #
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.BACKBONE.FREEZE_AT = 3

# ------------- RPN -------------- #
_CC.MODEL.RPN.FREEZE = False
_CC.MODEL.RPN.ENABLE_DECOUPLE = False
_CC.MODEL.RPN.BACKWARD_SCALE = 1.0

# ------------- ROI -------------- #
_CC.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.ROI_HEADS.ENABLE_DECOUPLE = False
_CC.MODEL.ROI_HEADS.BACKWARD_SCALE = 1.0
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
_CC.MODEL.ROI_HEADS.CLS_DROPOUT = False
_CC.MODEL.ROI_HEADS.DROPOUT_RATIO = 0.8
_CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # for faster
_CC.MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE = "CE"  # "CE" | "DC"

# ------------- TEST ------------- #
_CC.TEST.PCB_ENABLE = False
_CC.TEST.PCB_MODELTYPE = 'resnet'             # res-like
_CC.TEST.PCB_MODELPATH = ""
_CC.TEST.PCB_ALPHA = 0.50
_CC.TEST.PCB_UPPER = 1.0
_CC.TEST.PCB_LOWER = 0.05

# ------------- CASC ------------ #
_CC.TEST.CASC_ENABLE = False
_CC.TEST.CASC_MODELPATH = ""
_CC.TEST.CASC_ALPHA = 0.70
_CC.TEST.CASC_UPPER = 1.0
_CC.TEST.CASC_LOWER = 0.05
# ---------- SPPF (Spatial Pyramid Pooling - Fast, YOLOv5) ---------- #
_CC.MODEL.ROI_HEADS.SPPF_ENABLE = False  # only activate during fine-tuning

# ------------- TEF (Text-guided Feature Modulation / FiLM) ------ #
_CC.MODEL.ROI_HEADS.TFE_ENABLE = False
_CC.MODEL.ROI_HEADS.TFE_TEXT_FEATURES_PATH = "text_features/voc_all_classes.pt"
_CC.MODEL.ROI_HEADS.TFE_TEXT_DIM = 512
_CC.MODEL.ROI_HEADS.TFE_FREEZE_ALIGN = False

# ------------- TCC (Text-Conditioned Classifier) --------------- #
_CC.MODEL.ROI_HEADS.TCC_ENABLE = False
_CC.MODEL.ROI_HEADS.TCC_TEXT_FEATURES_PATH = ""
_CC.MODEL.ROI_HEADS.TCC_SCALE_INIT = 20.0

# ------------ Other ------------- #
_CC.SOLVER.WEIGHT_DECAY = 5e-5
_CC.MUTE_HEADER = True

# ------------- CGCL -------------- #
_CC.CGCL = CN()
_CC.CGCL.ENABLE = False
_CC.CGCL.SHOTS = 1
_CC.CGCL.CONTAINER = 2
_CC.CGCL.FEATURE_DIM = 128
_CC.CGCL.KNOWLEDGE_MATRIX = None
_CC.CGCL.TAU = 0.2
_CC.CGCL.COEF = 1.0