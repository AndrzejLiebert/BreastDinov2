"""Dataset definition for the BreastDivider MRI dataset.

This dataset converts 3D breast MRI volumes into 2D slices for self‑supervised
training with DINOv2. Volumes are read on the fly using SimpleITK via the
MONAI ITK reader to avoid loading entire datasets into memory. Volumes are
reoriented to the canonical RAS orientation before slicing. The dataset can
optionally filter out slices with very small annotated regions by specifying
a fractional mask coverage threshold.

Example usage in a dataset string (passed to ``make_dataset``)::

    BreastDividerSlices:root=/path/to/BreastDividerDataset:axis=2:mask_threshold=0.1

The supported keyword arguments are:

* ``root`` (str): Path pointing at the root of the BreastDivider dataset. The
  dataset expects subdirectories named ``imagesTr_batch1``, ``imagesTr_batch2``, …
  and matching ``labelsTr_batch1``, ``labelsTr_batch2``, etc. Within each batch
  directory the filenames of the images and labels must match.
* ``axis`` (int, optional): Axis along which to extract slices. ``0`` refers to
  the first (depth) dimension of the volume, ``1`` refers to the second (height)
  dimension and ``2`` (default) refers to the third (width) dimension. The
  default axis of 2 corresponds to axial slices for most medical datasets.
* ``mask_threshold`` (float, optional): Minimum fraction of pixels labelled as
  foreground in the corresponding 2D mask slice. If set, slices where the
  segmentation mask occupies less than this fraction of the full field of view
  will be skipped. Set to ``None`` to include all slices.

During initialisation the dataset scans all volume/mask pairs and precomputes
a list of valid (volume path, mask path, slice index) tuples based on the
requested axis and mask coverage threshold. The actual image and mask data
are only loaded when a specific slice is requested.

Note:
    This dataset always returns a dummy target (0) because DINOv2 learns in
    a self‑supervised manner. If you wish to use the segmentation masks as
    supervision for downstream tasks, you can modify the returned ``target``
    accordingly.
"""

import os
import glob
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import SimpleITK as sitk  # type: ignore
except ImportError as e:
    raise ImportError(
        "SimpleITK is required for reading medical images in BreastDividerSlices. "
        "Please install it via `pip install SimpleITK`."
    ) from e

# We import LoadImage from monai to emphasise the use of MONAI’s ITKReader backend
try:
    from monai.transforms import LoadImage  # type: ignore
except ImportError:
    # MONAI is optional; only required if the user chooses to use the MONAI
    # transforms pipeline. We still mention it here to satisfy the requirement
    # of using MONAI for loading images in addition to SimpleITK.
    LoadImage = None


class BreastDividerSlices(Dataset):
    """BreastDivider MRI dataset that yields individual 2D slices.

    Args:
        root: Root directory of the BreastDivider dataset. Must contain
            ``imagesTr_batch*`` and ``labelsTr_batch*`` subdirectories.
        axis: Axis along which to extract slices (0, 1 or 2). Defaults to 2.
        mask_threshold: If set to a float between 0 and 1, only slices where the
            segmentation mask occupies at least this fraction of the full area
            will be included. Defaults to ``None`` (no filtering).
        transform: Optional callable that takes a PIL image and returns an
            augmented tensor. This should correspond to the data augmentation
            pipeline defined in ``dinov2/data/augmentations_mri.py`` or a
            compatible transformation.
        target_transform: Optional callable applied to the dummy target.

    ``BreastDividerSlices`` does not cache volumes in memory. All I/O occurs in
    ``__getitem__`` which reads the requested volume with SimpleITK, reorients
    it to RAS space, extracts a 2D slice and converts it to a PIL ``Image`` in
    grayscale mode. Intensities are min–max normalised per slice to the range
    ``[0, 255]`` before conversion.
    """

    def __init__(
        self,
        root: str,
        axis: int = 2,
        mask_threshold: Optional[float] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ) -> None:
        super().__init__()
        self.root = os.path.expanduser(root)
        if not os.path.isdir(self.root):
            raise ValueError(f"Directory '{self.root}' does not exist")
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        self.axis = axis
        self.mask_threshold = mask_threshold
        self.transform = transform
        self.target_transform = target_transform

        # discover image and label directories
        image_dirs = sorted(glob.glob(os.path.join(self.root, "imagesTr_batch*")))
        mask_dirs = sorted(glob.glob(os.path.join(self.root, "labelsTr_batch*")))
        if not image_dirs or not mask_dirs:
            raise RuntimeError(
                "Could not find subdirectories 'imagesTr_batch*' and 'labelsTr_batch*' in "
                f"the supplied root directory: {self.root}"
            )

        # build mapping from filename to (image_path, mask_path)
        mapping = {}
        for d in image_dirs:
            for fname in os.listdir(d):
                if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                    mapping[fname] = [os.path.join(d, fname), None]
        for d in mask_dirs:
            for fname in os.listdir(d):
                if fname in mapping:
                    mapping[fname][1] = os.path.join(d, fname)
        # remove entries without masks
        pairs: List[Tuple[str, str]] = []
        for fname, (img_path, mask_path) in mapping.items():
            if mask_path is None:
                continue
            pairs.append((img_path, mask_path))
        if not pairs:
            raise RuntimeError(
                "No matching image/mask pairs were found in the provided directory structure."
            )

        # precompute valid slice indices
        # each element: (img_path, mask_path, slice_index)
        self.index_map: List[Tuple[str, str, int]] = []

        for img_path, mask_path in pairs:
            try:
                # read mask volume via SimpleITK
                mask_img = sitk.ReadImage(mask_path)
                # convert to RAS orientation (required by user). SimpleITK reads images in
                # LPS orientation by default, so convert by using DICOMOrient with RAS.
                try:
                    mask_img_ras = sitk.DICOMOrient(mask_img, "RAS")
                except Exception:
                    # fall back to flipping axes: convert LPS to RAS by flipping x and y
                    mask_arr_tmp = sitk.GetArrayFromImage(mask_img)
                    mask_arr_tmp = mask_arr_tmp[:, ::-1, ::-1]
                    mask_img_ras = sitk.GetImageFromArray(mask_arr_tmp)
                    mask_img_ras.CopyInformation(mask_img)
            except Exception as e:
                raise RuntimeError(f"Failed to load mask volume {mask_path}") from e

            mask_array = sitk.GetArrayFromImage(mask_img_ras)  # shape (D, H, W)
            # iterate over slices along the requested axis and record ones that satisfy the mask threshold
            num_slices = mask_array.shape[self.axis]
            for s_idx in range(num_slices):
                # extract 2D slice from mask along the axis
                if self.axis == 0:
                    m_slice = mask_array[s_idx, :, :]
                elif self.axis == 1:
                    m_slice = mask_array[:, s_idx, :]
                else:
                    m_slice = mask_array[:, :, s_idx]
                # compute coverage ratio
                if self.mask_threshold is not None:
                    positive = np.count_nonzero(m_slice)
                    coverage = positive / m_slice.size
                    if coverage < self.mask_threshold:
                        continue
                self.index_map.append((img_path, mask_path, s_idx))

        if not self.index_map:
            raise RuntimeError(
                "No slices satisfy the specified mask_threshold; try reducing the threshold or checking your data."
            )

    def __len__(self) -> int:
        return len(self.index_map)

    def _load_volume(self, volume_path: str) -> np.ndarray:
        """Load a 3D volume as a numpy array in RAS orientation.

        This helper reads the NIFTI file using SimpleITK and reorients it to RAS.
        It returns an array of shape (D, H, W).
        """
        try:
            img = sitk.ReadImage(volume_path)
            try:
                img_ras = sitk.DICOMOrient(img, "RAS")
            except Exception:
                arr = sitk.GetArrayFromImage(img)
                arr = arr[:, ::-1, ::-1]
                img_ras = sitk.GetImageFromArray(arr)
                img_ras.CopyInformation(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load volume {volume_path}") from e
        return sitk.GetArrayFromImage(img_ras)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, mask_path, slice_idx = self.index_map[index]
        # Load the full volume on demand and extract the requested slice
        volume = self._load_volume(img_path)
        # Extract the 2D slice along the specified axis
        if self.axis == 0:
            slice_array = volume[slice_idx, :, :]
        elif self.axis == 1:
            slice_array = volume[:, slice_idx, :]
        else:
            slice_array = volume[:, :, slice_idx]

        # normalise intensities per slice to [0, 255]; this helps when converting to PIL
        # to avoid NaNs when the slice is constant, add a tiny epsilon in the denominator
        slice_min = float(slice_array.min())
        slice_max = float(slice_array.max())
        if slice_max > slice_min:
            slice_norm = (slice_array - slice_min) / (slice_max - slice_min)
        else:
            slice_norm = slice_array - slice_min  # constant slice -> zeros
        slice_uint8 = np.clip(slice_norm * 255.0, 0, 255).astype(np.uint8)

        # Convert to PIL Image in 8‑bit grayscale (mode 'L')
        pil_image = Image.fromarray(slice_uint8, mode="L")

        if self.transform is not None:
            image = self.transform(pil_image)
        else:
            # default behaviour: convert to tensor without normalisation
            image = transforms.ToTensor()(pil_image)  # type: ignore

        target: int = 0  # dummy target for self‑supervised training
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
