import os
import cv2
import json
import torch
import logging
import detectron2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from detectron2.data import MetadataCatalog
from .archs import clip
import copy
from sklearn.metrics.pairwise import cosine_similarity
from sdcc.dataloader import build_detection_test_loader
from sdcc.evaluation.archs import resnet101

logger = logging.getLogger(__name__)


class SimpleMLP(nn.Module):
    """
    轻量化多层感知机：小权重初始化+缩小隐藏层规模，避免主导特征变换
    """

    def __init__(self, in_features, out_features, hidden_ratio=3, act_layer=nn.GELU, drop_rate=0.05):
        super().__init__()
        # 缩小隐藏层维度（hidden_ratio从2→3，隐藏层更小）
        hidden_features = max(in_features // hidden_ratio, out_features)  # 确保隐藏层不小于输出维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        # -------------------------- 关键优化：小权重初始化 --------------------------
        # fc1/fc2用Xavier初始化并缩放（0.5倍），避免初始权重过大
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.constant_(self.fc1.bias, 0.0)  # 偏置初始化为0，减少偏移
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)  # GELU激活更平缓，避免极端值
        x = self.drop(x)
        x = self.fc2(x)
        return x


class CrossSelfAttentionManual(nn.Module):
    """
    优化核心：残差缩放+弱正则化+小参数初始化，确保新增模块不主导特征输出
    """

    def __init__(self, dim=768, num_heads=4, drop_rate=0.05, ffn_hidden_ratio=3, residual_scale=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 注意力缩放因子（保留原逻辑）
        self.residual_scale = residual_scale  # FFN输出的残差缩放系数（核心控制参数）

        # ============ Self-Attention 模块（温和化改造） ============
        # LayerNorm：调大epsilon减少数值波动，避免过度归一化
        self.norm_self_attn = nn.LayerNorm(dim, eps=1e-6)
        self.q_self = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.proj_self = nn.Linear(dim, dim)  # 自注意力输出投影
        self.drop_self_attn = nn.Dropout(drop_rate)  # 降低Dropout速率

        # Self-Attention后的FFN（小权重+残差缩放）
        self.norm_self_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_self = SimpleMLP(dim, dim, hidden_ratio=ffn_hidden_ratio, drop_rate=drop_rate)

        # ============ Cross-Attention 模块（同逻辑温和化） ============
        self.norm_cross_attn = nn.LayerNorm(dim, eps=1e-6)
        self.q_cross = nn.Linear(dim, dim)
        self.k_cross = nn.Linear(dim, dim)
        self.v_cross = nn.Linear(dim, dim)
        self.proj_cross = nn.Linear(dim, dim)  # 交叉注意力输出投影
        self.drop_cross_attn = nn.Dropout(drop_rate)

        # Cross-Attention后的FFN（小权重+残差缩放）
        self.norm_cross_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_cross = SimpleMLP(dim, dim, hidden_ratio=ffn_hidden_ratio, drop_rate=drop_rate)

        # 最终融合的Dropout（保留原逻辑，速率同步降低）
        self.drop = nn.Dropout(drop_rate)

        # -------------------------- 关键优化：注意力层参数初始化 --------------------------
        # 所有Q/K/V/投影层用小权重初始化，避免初始特征偏移
        for module in [self.q_self, self.k_self, self.v_self, self.proj_self,
                       self.q_cross, self.k_cross, self.v_cross, self.proj_cross]:
            nn.init.xavier_uniform_(module.weight, gain=0.3)  # 增益0.3（更小权重）
            nn.init.constant_(module.bias, 0.0)  # 偏置置0

    def forward(self, img_feat, text_feat):
        """
        img_feat: (N, dim) - 图像特征（核心特征，保留主导地位）
        text_feat: (C, dim) - 文本特征（辅助特征，不主导）
        """
        N, _ = img_feat.shape
        C, _ = text_feat.shape
        H = self.num_heads

        # ----------------- 1. Self-Attention（弱干预逻辑） -----------------
        # Step1: LayerNorm（温和归一化）→ 注意力计算（保留原逻辑）
        x_sa = self.norm_self_attn(img_feat)  # 仅对输入做轻微归一化，不改变整体分布
        q = self.q_self(x_sa).view(N, H, self.head_dim).transpose(0, 1)
        k = self.k_self(x_sa).view(N, H, self.head_dim).transpose(0, 1)
        v = self.v_self(x_sa).view(N, H, self.head_dim).transpose(0, 1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # 移除原代码中不必要的 "/2"，保持标准注意力逻辑

        # Step2: 投影→Dropout→残差连接（无缩放，保留自注意力核心结果）
        sa_output = (attn @ v).transpose(0, 1).contiguous().view(N, self.dim)
        sa_output = self.proj_self(sa_output)
        sa_output = self.drop_self_attn(sa_output)
        sa_intermediate = img_feat + sa_output  # 自注意力结果直接残差，确保其作用

        # Step3: FFN（弱干预）→ 残差缩放（核心控制：FFN输出仅占10%）
        sa_ffn_output = self.norm_self_ffn(sa_intermediate)
        sa_ffn_output = self.ffn_self(sa_ffn_output)  # 小权重FFN，输出幅度小
        sa_out = sa_intermediate + self.residual_scale * sa_ffn_output  # FFN贡献仅10%

        # ----------------- 2. Cross-Attention（同弱干预逻辑） -----------------
        # Step1: LayerNorm→注意力计算（文本特征不做Norm，避免破坏预训练分布）
        x_ca = self.norm_cross_attn(sa_out)
        q_c = self.q_cross(x_ca).view(N, H, self.head_dim).transpose(0, 1)
        k_c = self.k_cross(text_feat).view(C, H, self.head_dim).transpose(0, 1)  # 文本特征直接用
        v_c = self.v_cross(text_feat).view(C, H, self.head_dim).transpose(0, 1)

        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.scale
        attn_c = F.softmax(attn_c, dim=-1)  # 移除原代码中不必要的 "/2"

        # Step2: 投影→Dropout→残差连接（无缩放，保留交叉注意力核心结果）
        ca_output = (attn_c @ v_c).transpose(0, 1).contiguous().view(N, self.dim)
        ca_output = self.proj_cross(ca_output)
        ca_output = self.drop_cross_attn(ca_output)
        ca_intermediate = sa_out + ca_output  # 交叉注意力结果直接残差

        # Step3: FFN（弱干预）→ 残差缩放（FFN贡献仅10%）
        ca_ffn_output = self.norm_cross_ffn(ca_intermediate)
        ca_ffn_output = self.ffn_cross(ca_ffn_output)
        ca_out = ca_intermediate + self.residual_scale * ca_ffn_output  # 控制FFN影响

        # ----------------- 3. 最终融合（严格保留原逻辑主导） -----------------
        # 交叉注意力结果（ca_out）仅占10%，原始图像特征（img_feat）占90%，确保核心逻辑不变
        fused = img_feat + self.drop(0.05 * ca_out)
        fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)  # 保留原L2归一化

        return fused


# 改进为CASC
class CASC:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        # alpha 由脚本 CASC_ALPHA 控制：final = alpha*检测器分数 + (1-alpha)*校准分数
        self.alpha = self.cfg.TEST.CASC_ALPHA       # 推荐 0.7（检测器占 70%，校准占 30%）


        # final_calib = text_weight * 文本路分数 + visual_weight * 视觉路分数
        self.text_weight   = 1.0   # 文本路占比（1.0=仅文本，等价原始基线）
        self.visual_weight = 1.0-self.text_weight   # 视觉路占比（>0 时自动构建 CLIP 视觉原型）
        # ============================================================= #

        # ==================== CLIP 模型（文本路 + 视觉路共用同一编码器）==================== #
        local_clip_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ViT-L-14-336px.pt"
        )
        if os.path.exists(local_clip_path):
            logger.info("使用本地 CLIP 模型: %s", local_clip_path)
            self.clip_model, self.preprocess = clip.load(
                local_clip_path, device=self.device
            )
        else:
            logger.info("本地 CLIP 模型不存在，开始远程下载...")
            self.clip_model, self.preprocess = clip.load(
                "ViT-L/14@336px", device=self.device
            )
        self.clip_model.cuda().eval()

        self.clip_roi_pooler = ROIPooler(
            output_size=(336, 336), scales=(1,),
            sampling_ratio=0, pooler_type="ROIAlignV2",
        )

        # ==================== 构建资产 ==================== #
        self.dataloader = build_detection_test_loader(
            self.cfg, self.cfg.DATASETS.TRAIN[0]
        )
        self.exclude_cls = self.clsid_filter()
        self.class_vector = self.text_encode()
        # 只有视觉路权重 > 0 时才构建 CLIP 视觉原型，否则跳过节省时间
        if self.visual_weight > 0:
            self.vis_proto_matrix = self._build_clip_visual_proto_matrix()
        else:
            self.vis_proto_matrix = None

        logger.info(
            "CASC dual-path (CLIP-CLIP): text_weight=%.2f, visual_weight=%.2f, alpha=%.2f",
            self.text_weight, self.visual_weight, self.alpha,
        )
        print("=" * 60)
        print("CASC 校准配置:")
        print(f"  文本路权重:         {self.text_weight:.2f}")
        print(f"  视觉路权重:         {self.visual_weight:.2f}")
        print(f"  检测器分数占比 α:   {self.alpha:.2f}")
        print(f"  校准分数占比 1-α:   {1 - self.alpha:.2f}")
        print(f"  Softmax 竞争类别数: {self.class_vector.shape[0]}  "
              f"(真实 {self.num_real_classes} + 负向锚点 {self.class_vector.shape[0] - self.num_real_classes})")
        print("=" * 60)

    # ------------------------------------------------------------------ #
    #                       文本编码（CLIP 文本路）                        #
    # ------------------------------------------------------------------ #
    def text_encode(self):
        dsname = self.cfg.DATASETS.TEST[0]
        self.classes = copy.deepcopy(MetadataCatalog.get(dsname).thing_classes)
        novel_id = copy.deepcopy(
            MetadataCatalog.get(dsname).get("novel_dataset_id_to_contiguous_id")
        )
        if novel_id is not None:
            thing_id = copy.deepcopy(
                MetadataCatalog.get(dsname).thing_dataset_id_to_contiguous_id
            )
            self.class_mapper = {
                thing_id[k]: idx for idx, k in enumerate(novel_id.keys())
            }
        elif 'voc' in dsname:
            self.class_mapper = {k: idx for idx, k in enumerate(range(15, 20))}
        else:
            raise NotImplementedError("implement class mapper for this dataset")

        prompts = []
        self.classes.append('background')
        for idx, _class in enumerate(self.classes):
            if idx in self.exclude_cls:
                continue
            prompts.append(f"a photo of {_class}")

        self.num_real_classes = len(prompts)  # 真实类别数（Novel + background）

        # ====== 负向语义锚点（Negative Semantic Anchors）====== #
        # 这些提示词描述了常见的假阳性模式（背景纹理、模糊裁剪等）
        # 在 Softmax 竞争中充当"概率黑洞"，吸收假阳性框的置信度
        negative_prompts = [
            "a blurry photo with no clear object",
            "a photo of random texture and noise",
            "a cropped meaningless image patch",
            "a photo of cluttered background scene",
        ]

        negative_prompts = [
    "a blurry photo with no clear object",           # 1
    "a photo of random texture and noise",            # 2
    "a cropped meaningless image patch",              # 3
    "a photo of cluttered background scene",          # 4
    "a photo of an occluded and unrecognizable object",  # 5
    "a dark photo with heavy shadow and no subject",     # 6
]
        prompts.extend(negative_prompts)
        # ===================================================== #

        logger.info("Text prompts: %d real + %d negative anchors = %d total",
                     self.num_real_classes, len(negative_prompts), len(prompts))

        text_tokens = clip.tokenize(prompts).cuda()
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features.to(torch.float32)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------ #
    #          CLIP 视觉原型构建（与文本路同一 768 维特征空间）             #
    # ------------------------------------------------------------------ #
    def _build_clip_visual_proto_matrix(self):
        """
        遍历 support 集，用 CLIP 视觉编码器提取每个 GT 框特征，
        按 class_mapper 行序拼成原型矩阵 [C_novel, 768]，L2 归一化。
        与文本路使用完全相同的特征空间，融合有意义。
        """
        logger.info("Building CLIP visual prototypes from support set...")
        all_features, all_labels = [], []
        for index in range(len(self.dataloader.dataset)):
            inputs = [self.dataloader.dataset[index]]
            assert len(inputs) == 1
            img = cv2.imread(inputs[0]['file_name'])
            img_h = img.shape[0]
            ratio = img_h / inputs[0]['instances'].image_size[0]
            inputs[0]['instances'].gt_boxes.tensor = \
                inputs[0]['instances'].gt_boxes.tensor * ratio
            boxes = [inputs[0]['instances'].gt_boxes.to(self.device)]
            # 用与推理相同的 CLIP 视觉编码器提取特征
            features = self._extract_clip_features(img, boxes)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            all_features.append(features.cpu().data)
            all_labels.append(inputs[0]['instances'].gt_classes.cpu().data)

        all_features = torch.cat(all_features, dim=0)  # [M, 768]
        all_labels = torch.cat(all_labels, dim=0)       # [M]

        # 按类别分组取均值
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features[i].unsqueeze(0))

        prototypes_dict = {}
        for label in features_dict:
            feats = torch.cat(features_dict[label], dim=0)
            proto = torch.mean(feats, dim=0, keepdim=True)
            prototypes_dict[label] = proto / (proto.norm(dim=-1, keepdim=True) + 1e-8)

        # 按 class_mapper 行序拼成矩阵
        C_novel = len(self.class_mapper)
        mat = torch.zeros(C_novel, 768).to(self.device)
        for cid, row in self.class_mapper.items():
            if cid in prototypes_dict:
                mat[row] = prototypes_dict[cid].squeeze(0).to(self.device)

        logger.info("CLIP visual prototypes built: %d classes", len(prototypes_dict))
        return mat  # 已 L2 归一化，[C_novel, 768]

    # ------------------------------------------------------------------ #
    #                     CLIP ROI 特征提取                                #
    # ------------------------------------------------------------------ #
    def _extract_clip_features(self, img, boxes):
        """用 CLIP 视觉编码器提取 ROI 特征。"""
        img = img.transpose((2, 0, 1)).copy()   # copy() 确保连续内存，避免 view 共享
        img = torch.from_numpy(img).to(self.device)
        images = [img / 255.]
        images = ImageList.from_tensors(images, 0)
        box_features = self.clip_roi_pooler([images.tensor], boxes)
        if len(box_features) == 0:
            return torch.zeros(1, 768, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            conv_feature = self.clip_model.encode_image(box_features)
        return conv_feature.to(torch.float32)

    # ------------------------------------------------------------------ #
    #                     校准主逻辑（双路融合）                           #
    # ------------------------------------------------------------------ #
    def execute_calibration(self, inputs, dts):
        img = cv2.imread(inputs[0]['file_name'])

        ileft = (dts[0]['instances'].scores > self.cfg.TEST.CASC_UPPER).sum()
        iright = (dts[0]['instances'].scores > self.cfg.TEST.CASC_LOWER).sum()
        ileft = int(ileft.detach().cpu().numpy())
        iright = int(iright.detach().cpu().numpy())
        assert ileft <= iright

        idx = []
        pred_class_list = []
        for i in range(ileft, iright):
            pred_class = int(dts[0]['instances'].pred_classes[i])
            if pred_class in self.exclude_cls:
                continue
            if pred_class not in self.class_mapper:
                continue
            pred_class_list.append(self.class_mapper[pred_class])
            idx.append(i)

        if not idx:
            return dts

        idx = np.array(idx)
        pred_class_list = np.array(pred_class_list)
        boxes = [dts[0]['instances'].pred_boxes[idx]]

        # 提取 CLIP 视觉特征（两路共用同一特征，保证特征空间一致）
        clip_feat = self._extract_clip_features(img, boxes)
        clip_feat = clip_feat / (clip_feat.norm(dim=-1, keepdim=True) + 1e-8)  # [N, 768]

        # ----- 文本路：clip_feat vs CLIP 文本嵌入 ----- #
        text_sim = clip_feat @ self.class_vector.T                    # [N, C_all]
        t_alpha = F.relu(text_sim) + 1e-3
        t_dirichlet = t_alpha / t_alpha.sum(dim=1, keepdim=True)
        t_softmax = F.softmax(text_sim.float() * 100, dim=1)
        text_score = 0.95 * t_softmax + 0.05 * t_dirichlet           # [N, C_all]

        # ----- 取预测类列（文本路） ----- #
        text_pred = text_score[range(len(idx)), pred_class_list]      # [N]

        # ----- 视觉原型路：clip_feat vs CLIP 视觉原型（同一 768 维空间）----- #
        if self.visual_weight > 0 and self.vis_proto_matrix is not None:
            vis_sim = clip_feat @ self.vis_proto_matrix.T             # [N, C_novel]
            v_alpha = F.relu(vis_sim) + 1e-3
            v_dirichlet = v_alpha / v_alpha.sum(dim=1, keepdim=True)
            v_softmax = F.softmax(vis_sim.float() * 100, dim=1)
            vis_score = 0.95 * v_softmax + 0.05 * v_dirichlet        # [N, C_novel]
            vis_pred = vis_score[range(len(idx)), pred_class_list]    # [N]
            calib_score = self.text_weight * text_pred + self.visual_weight * vis_pred
        else:
            calib_score = text_pred

        dts[0]['instances'].scores[idx] = (
            dts[0]['instances'].scores[idx] * self.alpha +
            calib_score * (1 - self.alpha)
        )

        return dts

    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                               66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids




class PrototypicalCalibrationBlock:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.alpha = self.cfg.TEST.PCB_ALPHA

        self.imagenet_model = self.build_model()
        self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")
        self.prototypes = self.build_prototypes()

        self.exclude_cls = self.clsid_filter()

    def build_model(self):
        logger.info("Loading ImageNet Pre-train Model from {}".format(self.cfg.TEST.PCB_MODELPATH))
        if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    def build_prototypes(self):

        all_features, all_labels = [], []
        for index in range(len(self.dataloader.dataset)):
            inputs = [self.dataloader.dataset[index]]
            assert len(inputs) == 1
            # load support images and gt-boxes
            img = cv2.imread(inputs[0]['file_name'])  # BGR
            img_h, img_w = img.shape[0], img.shape[1]
            ratio = img_h / inputs[0]['instances'].image_size[0]
            inputs[0]['instances'].gt_boxes.tensor = inputs[0]['instances'].gt_boxes.tensor * ratio
            boxes = [x["instances"].gt_boxes.to(self.device) for x in inputs]

            # extract roi features
            features = self.extract_roi_features(img, boxes)
            all_features.append(features.cpu().data)

            gt_classes = [x['instances'].gt_classes for x in inputs]
            all_labels.append(gt_classes[0].cpu().data)

        # concat
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        assert all_features.shape[0] == all_labels.shape[0]

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features[i].unsqueeze(0))

        prototypes_dict = {}
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)

        return prototypes_dict

    def extract_roi_features(self, img, boxes):
        """
        :param img:
        :param boxes:
        :return:
        """

        mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
        std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255. - mean) / std]
        images = ImageList.from_tensors(images, 0)
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW

        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)

        activation_vectors = self.imagenet_model.fc(box_features)

        return activation_vectors

    def execute_calibration(self, inputs, dts):

        img = cv2.imread(inputs[0]['file_name'])

        ileft = (dts[0]['instances'].scores > self.cfg.TEST.PCB_UPPER).sum()
        iright = (dts[0]['instances'].scores > self.cfg.TEST.PCB_LOWER).sum()
        assert ileft <= iright
        boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]

        features = self.extract_roi_features(img, boxes)

        for i in range(ileft, iright):
            tmp_class = int(dts[0]['instances'].pred_classes[i])
            if tmp_class in self.exclude_cls:
                continue
            tmp_cos = cosine_similarity(features[i - ileft].cpu().data.numpy().reshape((1, -1)),
                                        self.prototypes[tmp_class].cpu().data.numpy())[0][0]
            dts[0]['instances'].scores[i] = dts[0]['instances'].scores[i] * self.alpha + tmp_cos * (1 - self.alpha)
        return dts

    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                               66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output






