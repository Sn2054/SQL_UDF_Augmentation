import unittest

from torch import nn

from models.zero_shot_models.zero_shot_model import ZeroShotModel


class ZeroShotModelAugmentationSwitchTest(unittest.TestCase):
    def make_model(self, with_augmentor=True):
        model = ZeroShotModel.__new__(ZeroShotModel)
        nn.Module.__init__(model)
        model.graph_augmentor = nn.Identity() if with_augmentor else None
        if model.graph_augmentor is not None:
            model.graph_augmentor.last_coarse_fine_loss = object()
        model.augmentation_enabled = with_augmentor
        return model

    def test_runtime_switch_disables_and_reenables_existing_augmentor(self):
        model = self.make_model()

        model.set_augmentation_enabled(False)
        self.assertFalse(model.augmentation_enabled)
        self.assertIsNone(model.graph_augmentor.last_coarse_fine_loss)

        model.set_augmentation_enabled(True)
        self.assertTrue(model.augmentation_enabled)

    def test_runtime_switch_has_no_effect_without_training_augmentation(self):
        model = self.make_model(with_augmentor=False)

        model.set_augmentation_enabled(True)

        self.assertFalse(model.augmentation_enabled)


if __name__ == "__main__":
    unittest.main()
