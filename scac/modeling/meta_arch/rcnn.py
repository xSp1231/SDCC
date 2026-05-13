import torch
import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from scac.modeling.roi_heads import build_roi_heads

__all__ = ["GeneralizedRCNN"]

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        fs_class = [x.get("fs_class", []) for x in batched_inputs]
        # print("666666")

        # print("==== DEBUG: 检查微调阶段的真实框数量 ====")
        # for i, instances in enumerate(gt_instances):
        #     num_boxes = len(instances.gt_boxes)
        #     classes = instances.gt_classes.tolist()
        #     print(f"  Image {i} in batch: {num_boxes} GT boxes, Classes: {classes}")
        # print("===========================================")
        """
        ==== DEBUG: 检查微调阶段的真实框数量 ====
          Image 0 in batch: 1 GT boxes, Classes: [19]
          Image 1 in batch: 1 GT boxes, Classes: [18]
          Image 2 in batch: 1 GT boxes, Classes: [14]
          Image 3 in batch: 1 GT boxes, Classes: [7]
          Image 4 in batch: 1 GT boxes, Classes: [11]
          Image 5 in batch: 1 GT boxes, Classes: [2]
          Image 6 in batch: 1 GT boxes, Classes: [2]
          Image 7 in batch: 1 GT boxes, Classes: [5]
          Image 8 in batch: 1 GT boxes, Classes: [16]
          Image 9 in batch: 1 GT boxes, Classes: [5]
          Image 10 in batch: 1 GT boxes, Classes: [2]
          Image 11 in batch: 1 GT boxes, Classes: [15]
          Image 12 in batch: 1 GT boxes, Classes: [8]
          Image 13 in batch: 1 GT boxes, Classes: [13]
          Image 14 in batch: 1 GT boxes, Classes: [1]
          Image 15 in batch: 1 GT boxes, Classes: [6]
        """
        """
        微调阶段确实是“单框”输入
        无论原始的 VOC 图片里到底有多热闹（可能有 5 个人、3 辆车、1 只狗），在传进模型的时候，每张图片真的只配给了一个真实框（Ground Truth box）。
        这也彻底解释了为什么在 roi_heads.py 的代码里，提取原型特征时敢于大胆地写 x.gt_classes[0] ——因为列表的长度绝对是 1，永远不可能有 gt_classes[1] 让你去取。
        结论二：Batch Size = 16 确实能凑齐大量不同类别
        你看看这 16 张图片的 Classes 列表（也就是类别 ID）：
        [19, 18, 14, 7, 11, 2, 2, 5, 16, 5, 2, 15, 8, 13, 1, 6]
        虽然有一点点重复（比如类别 2 出现了 3 次，类别 5 出现了 2 次），但在这个只有 16 张图的 Batch 里，总共凑齐了 13 种截然不同的物体类别！
        这就是为什么之前我说即使是 1-shot，也千万别把 Batch Size 设成 1。如果你只设成 1，那模型这次只能看到类别 19，下次只能看到类别 18。而在设为 16 时，模型在一个 iteration 里就能同时学到 13 种物体的特征，这种“全局上帝视角”对更新参数来说要平滑得多。
        结论三：为 CGCL 提供了完美的“负样本温床”
        正因为这 16 个框涵盖了十几种类别，当 _extract_proposal_feature 去提取出对应的候选框（proposals）时，它们彼此之间就是天然且丰富的负样本（Negative Samples）。在计算对比损失的时候，代表类别 19 的特征和代表类别 2 的特征就可以在同一个计算图里被互相推开，从而使特征空间边界更清晰。
        
        """
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances, fs_class)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None, fs_class=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances, fs_class)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
