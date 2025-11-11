"""Prototypical network model definitions."""

import torch
from torch import nn
from torchvision.models import resnet18


class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for few-shot classification."""
    
    def __init__(self, backbone: nn.Module, freeze_backbone: bool = True) -> None:
        """Initialize network.
        
        Args:
            backbone: Feature extraction network
            freeze_backbone: Freeze backbone weights
        """
        super().__init__()
        self.backbone = backbone
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self) -> None:
        """Freeze layers except layer4 and final."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = 'layer4' in name or 'fc' in name
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification scores.
        
        Args:
            support_images: [n_support, C, H, W]
            support_labels: [n_support]
            query_images: [n_query, C, H, W]
            
        Returns:
            Classification scores [n_query, n_way]
        """
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)
        
        n_way = len(torch.unique(support_labels))
        
        prototypes = []
        for label in range(n_way):
            class_mask = (support_labels == label)
            prototypes.append(z_support[class_mask].mean(dim=0))
        
        z_proto = torch.stack(prototypes, dim=0)
        dists = torch.cdist(z_query, z_proto)
        
        return -dists


def create_backbone(backbone_name: str, pretrained: bool) -> nn.Module:
    """Create backbone network.
    
    Args:
        backbone_name: Name of backbone architecture
        pretrained: Use pretrained weights
        
    Returns:
        Backbone with identity final layer
    """
    if backbone_name != "resnet18":
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    backbone = resnet18(pretrained=pretrained)
    backbone.fc = nn.Identity()
    return backbone

