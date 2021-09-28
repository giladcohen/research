import logging
from typing import List
from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class PyTorchClassifierSpecific(PyTorchClassifier):  # lgtm [py/missing-call-to-init]
    def __init__(self, *args, **kwargs):
        self.field = kwargs.pop('field')
        super().__init__(*args, **kwargs)

    def _make_model_wrapper(self, model: "torch.nn.Module") -> "torch.nn.Module":
        # Try to import PyTorch and create an internal class that acts like a model wrapper extending torch.nn.Module
        try:
            import torch.nn as nn

            # Define model wrapping class only if not defined before
            if not hasattr(self, "_model_wrapper"):

                class ModelWrapper(nn.Module):
                    """
                    This is a wrapper for the input model.
                    """

                    import torch  # lgtm [py/repeated-import]

                    def __init__(self, model: torch.nn.Module, field: str):
                        """
                        Initialization by storing the input model.

                        :param model: PyTorch model. The forward function of the model must return the logit output.
                        """
                        super().__init__()
                        self._model = model
                        self.field = field

                    # pylint: disable=W0221
                    # disable pylint because of API requirements for function
                    def forward(self, x):
                        """
                        This is where we get outputs from the input model.

                        :param x: Input data.
                        :type x: `torch.Tensor`
                        :return: a list of output layers, where the last 2 layers are logit and final outputs.
                        :rtype: `list`
                        """
                        # pylint: disable=W0212
                        # disable pylint because access to _model required
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            for _, module_ in self._model._modules.items():
                                x = module_(x)
                                result.append(x)

                        elif isinstance(self._model, nn.Module):
                            x = self._model(x)[self.field]
                            result.append(x)

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")

                        return result

                    @property
                    def get_layers(self) -> List[str]:
                        """
                        Return the hidden layers in the model, if applicable.

                        :return: The hidden layers in the model, input and output layers excluded.

                        .. warning:: `get_layers` tries to infer the internal structure of the model.
                                     This feature comes with no guarantees on the correctness of the result.
                                     The intended order of the layers tries to match their order in the model, but this
                                     is not guaranteed either. In addition, the function can only infer the internal
                                     layers if the input model is of type `nn.Sequential`, otherwise, it will only
                                     return the logit layer.
                        """
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            # pylint: disable=W0212
                            # disable pylint because access to _modules required
                            for name, module_ in self._model._modules.items():  # type: ignore
                                result.append(name + "_" + str(module_))

                        elif isinstance(self._model, nn.Module):
                            result.append("final_layer")

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")
                        logger.info(
                            "Inferred %i hidden layers on PyTorch classifier.",
                            len(result),
                        )

                        return result

                # Set newly created class as private attribute
                self._model_wrapper = ModelWrapper

            # Use model wrapping class to wrap the PyTorch model received as argument
            return self._model_wrapper(model, self.field)

        except ImportError:
            raise ImportError("Could not find PyTorch (`torch`) installation.") from ImportError

    def __repr__(self):
        repr_ = (
                "%s(model=%r, loss=%r, optimizer=%r, input_shape=%r, nb_classes=%r, channels_first=%r, "
                "clip_values=%r, preprocessing_defences=%r, postprocessing_defences=%r, preprocessing=%r, field=%r)"
                % (
                    self.__module__ + "." + self.__class__.__name__,
                    self._model,
                    self._loss,
                    self._optimizer,
                    self._input_shape,
                    self.nb_classes,
                    self.channels_first,
                    self.clip_values,
                    self.preprocessing_defences,
                    self.postprocessing_defences,
                    self.preprocessing,
                    self.field
                )
        )

        return repr_
