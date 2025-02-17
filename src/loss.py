import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, alpha: float=1.0, symmetric: bool=True,):
        """
        Initialize our custom loss module.

        Parameters
        ----------
        alpha
            Weighting of the KL terms in the loss.
        symmetric
            Whether or not to include 2 KL divergences for symmetry.
        """
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.symmetric = symmetric

    def forward(self, 
        output1: torch.Tensor, 
        output2: torch.Tensor, 
        target: torch.Tensor,
        ) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Parameters
        ----------
        output1 
            Output logits for the 1st version of the neighbor patches.
        output2
            Output logits for the 2nd version of the neighbor patches.
        target
            Ground truth labels (1-hot vectors).

        Returns:
        -------
        torch.Tensor)
            Loss between outputs and target.
        """
        # cross entropies
        loss = F.cross_entropy(output1, target) + F.cross_entropy(output2, target)

        # define output probability distributions (kl_div expects the input to be in log space)
        log_prob1 = F.log_softmax(output1, dim=1)
        log_prob2 = F.log_softmax(output2, dim=1)

        # KL divergences
        kl = F.kl_div(log_prob1, log_prob2, log_target=True, reduction="batchmean")
        if self.symmetric:
            kl += F.kl_div(log_prob2, log_prob1, log_target=True, reduction="batchmean")
        loss += self.alpha * kl

        return loss
        