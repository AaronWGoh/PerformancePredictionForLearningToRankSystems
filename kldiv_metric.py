import torch


def check_same_shape(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Raises an error if predictions and targets do not have same dimensions
    """
    if predictions.shape != targets.shape:
        raise RuntimeError(
            "Predictions and Targets are expected to have same dimensions"
        )


def convert_to_tensor(obj):
    """
    Converts to Tensor if given object is not a Tensor.
    """
    if not isinstance(obj, torch.Tensor):
        obj = torch.Tensor(obj)

    return obj


class KLDivergence:
    """
    Computes Kullback-Leibler divergence metric between `y_true` and `y_pred`.
    Args:
        y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
        y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]
        epsilon: Fuzz factor to avoid division by zero. default: `1e-10`
    Returns:
        Tensor of Kullback-Leibler divergence metric
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)

        check_same_shape(y_pred, y_true)

        y_pred = torch.clamp(y_pred, self.epsilon, 1)
        y_true = torch.clamp(y_true, self.epsilon, 1)
        kld = torch.sum(y_true * torch.log(y_true / y_pred), axis=-1)
        return torch.mean(kld)
