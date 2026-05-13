import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

import numpy as np


class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """

    def __init__(self, dim_in=2048, feat_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized


class CGCLLoss(nn.Module):
    """
    Loss function of the class-guided contrastive learning (CGCL)
    """

    def __init__(self, knowledge_martix_path, coefficient=1, tau=0.2):
        super().__init__()
        self.knowledge_matrix = np.load(knowledge_martix_path)
        # self.knowledge_matrix = (self.knowledge_matrix - 0.5) * 2   # 新增
        np.fill_diagonal(self.knowledge_matrix, 1.0)  # 对角线设置为1

        # --- [新增] 输出知识矩阵 ---
        print("\n" + "=" * 40)
        print(f"Loaded Knowledge Matrix from:\n{knowledge_martix_path}")
        print(f"Matrix shape: {self.knowledge_matrix.shape}")
        print("Matrix content:")
        # 因为矩阵可能有 20x20 这么大，直接 print 可能换行很难看
        # 这里用 numpy 的参数让它显示得更紧凑一些
        with np.printoptions(precision=3, suppress=True, linewidth=150):
            print(self.knowledge_matrix)
        print("=" * 40 + "\n", flush=True)
        # ---------------------------
        """
        Matrix shape: (20, 20)
        Matrix content:
        [[0.871 0.691 0.726 0.66  0.726 0.719 0.669 0.64  0.698 0.625 0.633 0.635 0.634 0.774 0.68  0.718 0.51  0.677 0.514 0.549]
         [0.691 0.873 0.685 0.667 0.688 0.723 0.673 0.647 0.693 0.646 0.667 0.672 0.657 0.7   0.652 0.646 0.52  0.817 0.528 0.544]
         [0.726 0.685 0.832 0.689 0.68  0.69  0.67  0.731 0.679 0.719 0.708 0.685 0.665 0.684 0.661 0.519 0.703 0.514 0.704 0.522]
         [0.66  0.667 0.689 0.902 0.69  0.674 0.671 0.652 0.704 0.644 0.636 0.676 0.69  0.659 0.679 0.582 0.514 0.569 0.52  0.532]
         [0.726 0.688 0.68  0.69  0.858 0.812 0.68  0.641 0.69  0.628 0.631 0.648 0.648 0.812 0.7   0.687 0.513 0.682 0.517 0.561]
         [0.719 0.723 0.69  0.674 0.812 0.885 0.687 0.654 0.693 0.642 0.646 0.664 0.648 0.782 0.689 0.695 0.52  0.725 0.528 0.575]
         [0.669 0.673 0.67  0.671 0.68  0.687 0.855 0.652 0.808 0.661 0.645 0.693 0.672 0.668 0.723 0.672 0.524 0.574 0.534 0.855]
         [0.64  0.647 0.731 0.652 0.641 0.654 0.652 0.853 0.643 0.831 0.814 0.677 0.658 0.636 0.629 0.537 0.908 0.528 0.775 0.542]
         [0.698 0.693 0.679 0.704 0.69  0.693 0.808 0.643 0.878 0.649 0.641 0.673 0.696 0.697 0.708 0.715 0.536 0.596 0.549 0.665]
         [0.625 0.646 0.719 0.644 0.628 0.642 0.661 0.831 0.649 0.904 0.832 0.684 0.639 0.624 0.629 0.523 0.984 0.517 0.797 0.526]
         [0.633 0.667 0.708 0.636 0.631 0.646 0.645 0.814 0.641 0.832 0.895 0.681 0.638 0.636 0.617 0.522 0.915 0.517 0.784 0.525]
         [0.635 0.672 0.685 0.676 0.648 0.664 0.693 0.677 0.673 0.684 0.681 0.846 0.649 0.632 0.645 0.528 0.669 0.522 0.67  0.583]
         [0.634 0.657 0.665 0.69  0.648 0.648 0.672 0.658 0.696 0.639 0.638 0.649 0.921 0.636 0.64  0.533 0.51  0.514 0.514 0.521]
         [0.774 0.7   0.684 0.659 0.812 0.782 0.668 0.636 0.697 0.624 0.636 0.632 0.636 0.892 0.68  0.712 0.5   0.663 0.5   0.543]
         [0.68  0.652 0.661 0.679 0.7   0.688 0.723 0.629 0.708 0.629 0.617 0.645 0.64  0.68  0.893 0.655 0.513 0.611 0.518 0.569]
         [0.718 0.646 0.519 0.582 0.687 0.695 0.672 0.537 0.715 0.523 0.522 0.528 0.533 0.712 0.655 1.    0.52  0.644 0.527 0.608]
         [0.51  0.52  0.703 0.514 0.513 0.52  0.524 0.908 0.536 0.984 0.915 0.669 0.51  0.5   0.513 0.52  1.    0.515 0.792 0.523]
         [0.677 0.817 0.514 0.569 0.682 0.725 0.574 0.528 0.596 0.517 0.517 0.522 0.514 0.663 0.611 0.644 0.515 1.    0.521 0.534]
         [0.514 0.528 0.704 0.52  0.517 0.528 0.534 0.775 0.549 0.797 0.784 0.67  0.514 0.5   0.518 0.527 0.792 0.521 1.    0.532]
         [0.549 0.544 0.522 0.532 0.561 0.575 0.855 0.542 0.665 0.526 0.525 0.583 0.521 0.543 0.569 0.608 0.523 0.534 0.532 1.   ]]
            """
        self.coefficient = coefficient
        self.tau = tau
        # self.iou_threshold = 0.7



    def CGCL_loss(self, prototype_features, prototype_classes, object_features, object_labels):
        # Compute the Cosine distance
        x_norm = torch.nn.functional.normalize(object_features, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(prototype_features, p=2, dim=1)
        similarity = torch.mm(x_norm, y_norm.t())  # torch.Size([N2,N1])
        similarity = similarity - similarity.detach().max()

        # Judge the positive pair and negative pair
        pos_matrix = (prototype_classes == object_labels.unsqueeze(1))  # torch.Size([N2,N1])

        # Compute the knowledge matrix zeta
        index_x = object_labels.unsqueeze(1).repeat(1, prototype_classes.shape[0]).reshape(
            -1)  # Size: torch.Size([N2*N1])
        index_y = prototype_classes.repeat(object_labels.shape[0], 1).reshape(-1)

        zeta = self.knowledge_matrix[index_x.cpu().numpy(), index_y.cpu().numpy()]
        zeta = zeta.reshape(object_labels.shape[0], prototype_classes.shape[0])

        similarity = torch.exp(torch.from_numpy(zeta).cuda() * similarity / self.tau)

        pos_similarity = torch.sum(similarity * pos_matrix.int(), dim=1)  # shape: torch.Size([N1])

        if pos_similarity.min() == 0:  # May cause error
            print("损失归为0")
            return torch.tensor(0.).cuda()

        neg_similarity = torch.sum(similarity, dim=1)  # shape: torch.Size([N1])  # 也就是正负样本和

        Loss = -torch.mean(torch.log(pos_similarity / neg_similarity))
        return Loss

    def forward(self, prototype_features, prototype_classes, object_features, object_labels):
        """CGCL loss

        Args:
            prototype_features (torch.Size([N1, FEATURE_DIM])): Feature embeddings of the prototypes.
            prototype_classes (List): Labels of the prototypes.
            object_features (torch.Size([N2, FEATURE_DIM])): Feature embeddings of the proposals.
            object_labels (List): Labels of the proposals.

        Returns:
            Loss: _description_
        """
        Loss = self.CGCL_loss(prototype_features, prototype_classes, object_features, object_labels)

        return self.coefficient * Loss
