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


class MeanSquaredError:
    def __call__(self, tensor1, tensor2):
        """
        Arguments
        ---------
        x : torch.Tensor
        y : torch.Tensor
        """
        tensor1 = convert_to_tensor(tensor1)
        tensor2 = convert_to_tensor(tensor2)

        check_same_shape(tensor1, tensor2)
        return torch.mean((tensor1 - tensor2) ** 2)
