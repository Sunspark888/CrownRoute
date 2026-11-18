"""CPU-only comparison of the unweighted CrownRoute mask loss and torchvision's reference."""

import torch
from torchvision.models.detection.roi_heads import (
    maskrcnn_loss as torchvision_mask_loss,
)
from loss import weighted_maskrcnn_loss


def main():
    torch.manual_seed(42)
    masks = torch.zeros((2, 32, 32), dtype=torch.uint8)
    masks[0, 2:12, 3:15] = 1
    masks[1, 15:30, 16:31] = 1
    proposals = [torch.tensor([[3, 2, 15, 12], [16, 15, 31, 30]], dtype=torch.float32)]
    labels, matches = [torch.ones(2, dtype=torch.int64)], [torch.tensor([0, 1])]
    a = torch.randn(2, 2, 14, 14, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    source = torchvision_mask_loss(a, proposals, [masks], labels, matches)
    candidate = weighted_maskrcnn_loss(b, proposals, [masks], labels, matches)
    torch.testing.assert_close(source, candidate)
    source.backward()
    candidate.backward()
    torch.testing.assert_close(a.grad, b.grad)
    print("Matched unweighted ROI mask loss and gradients on two synthetic instances.")


if __name__ == "__main__":
    main()
