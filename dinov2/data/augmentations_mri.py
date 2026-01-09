"""Augmentation pipeline for MRI slices used in DINOv2 training.

The default DINOv2 augmentation pipeline is tuned for natural RGB images and
relies on colour jitter and solarisation. For single‑channel medical images
(e.g. breast MRI) these operations are either undefined or degrade the
representation quality. ``MRIDataAugmentationDINO`` provides a similar
multi‑crop augmentation pipeline adapted to 2D grayscale slices:

* Global and local crops with random resized crops and horizontal flips.
* Random brightness and contrast adjustments.
* Gaussian blurring as in the original DINO pipeline.
* Normalisation using a single mean and standard deviation for 1‑channel
  inputs (default 0.5, 0.5). The output tensors therefore have shape
  ``[1, H, W]``.

This augmentation class mirrors the interface of ``DataAugmentationDINO`` to
simplify integration into the existing training loop. The augmentation
hyper‑parameters (crop sizes and scales, number of local crops, etc.) should be
provided through the configuration file in the same way as for the standard
pipeline.
"""

import logging
from torchvision import transforms

from .transforms import GaussianBlur

logger = logging.getLogger("dinov2")


class MRIDataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size: int = 224,
        local_crops_size: int = 96,
        mean: float = 0.5,
        std: float = 0.5,
    ) -> None:
        """
        Args:
            global_crops_scale: Range of scale for the random resized crop of global crops.
            local_crops_scale: Range of scale for the random resized crop of local crops.
            local_crops_number: Number of local crops to generate.
            global_crops_size: Output size of global crops (square). Defaults to 224.
            local_crops_size: Output size of local crops (square). Defaults to 96.
            mean: Normalisation mean for 1‑channel images. Defaults to 0.5.
            std: Normalisation standard deviation for 1‑channel images. Defaults to 0.5.
        """
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using MRI data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"mean: {mean}")
        logger.info(f"std: {std}")
        logger.info("###################################")

        # geometric augmentations
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # brightness/contrast jitter (no hue or saturation for grayscale)
        color_jittering = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8
        )

        # additional global transformations
        global_transfo1_extra = GaussianBlur(p=1.0)
        global_transfo2_extra = transforms.Compose([
            GaussianBlur(p=0.1),
        ])  # omit solarisation for 1‑channel images
        local_transfo_extra = GaussianBlur(p=0.5)

        # normalisation to tensor
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean], std=[std]),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        """Apply the augmentation pipeline to a PIL image.

        Returns a dictionary with the same keys as ``DataAugmentationDINO``:
        ``global_crops`` (list of two tensors), ``global_crops_teacher`` (same
        list), ``local_crops`` (list of tensors) and ``offsets`` (empty tuple).
        """
        output = {}

        # global crops
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
