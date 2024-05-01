import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss implementation.
    Calculates the Dice Loss between the predicted input and the target.
    """

    def forward(self, input, target):
        """
        Calculates the forward pass of the Dice Loss.

        Args:
            input (torch.Tensor): Predicted input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Dice Loss value.
        """
        smooth = 1.
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss implementation.
    Calculates the IoU Loss between the predicted input and the target.
    """

    def forward(self, input, target):
        """
        Calculates the forward pass of the IoU Loss.

        Args:
            input (torch.Tensor): Predicted input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: IoU Loss value.
        """
        smooth = 1.
        intersection = (input * target).sum()
        total = (input + target).sum()
        union = total - intersection

        return 1 - ((intersection + smooth) / (union + smooth))
