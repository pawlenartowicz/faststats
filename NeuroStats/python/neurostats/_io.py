"""nilearn objects in, nilearn images out: ``import_from``, ``export_to`` and
``MaskedData``.

A masker, a mask image and an image are recognised by their attributes, so
nilearn is not imported to recognise anything. It is imported in one place:
to mask images through a mask image that came without a masker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import _arrays
from ._arrays import Array
from ._tfce import Domain


@dataclass(frozen=True, eq=False)
class MaskedData:
    """Brain data in the form the inference functions read.

    - ``y``: ``(n, V)``, one row per subject; column ``j`` is node ``j`` of
      ``domain``.
    - ``domain``: the ``Domain``.
    - ``origin``: the masker or mask image given as ``space``, or ``None``.
    """

    y: Array
    domain: Domain
    origin: object


def _is_image(obj) -> bool:
    return hasattr(obj, "affine") and hasattr(obj, "dataobj")


def _is_masker(obj) -> bool:
    return hasattr(obj, "transform") and hasattr(obj, "inverse_transform")


def _is_images(data) -> bool:
    """One image or one file name, or a list or tuple that starts with one."""
    if isinstance(data, (list, tuple)) and data:
        data = data[0]
    return _is_image(data) or isinstance(data, (str, os.PathLike))


def _mask_image(host, name: str):
    """The mask image of ``host``: ``host.mask_img_`` for a fitted masker,
    ``host`` itself for an image, ``None`` for anything else. ``name`` is the
    argument name used in the errors."""
    if _is_masker(host):
        if not hasattr(host, "mask_img_"):
            raise ValueError(f"{name} is a masker that is not fitted; call its fit() first")
        if not _is_image(host.mask_img_):
            raise ValueError(
                f"{name} is a masker whose mask_img_ is not an image "
                f"(got {type(host.mask_img_).__name__}); surface maskers are not supported"
            )
        return host.mask_img_
    return host if _is_image(host) else None


def import_from(data, space=None, *, conn: int | None = None) -> MaskedData:
    """Brain data and their space as a :class:`MaskedData`.

    Every inference function calls this on its ``data``, ``space`` and
    ``conn`` arguments. Call it yourself to mask once and run several
    inferences on the result.

    ``space`` and ``data``:

    - a ``Domain`` and an ``(n, V)`` array: used as they are; ``origin`` is
      ``None``;
    - a fitted masker and images: ``masker.transform(images)``;
    - a fitted masker and an ``(n, V)`` array: the array as it is;
    - a mask image and images: masked with a nilearn ``NiftiMasker`` built on
      that mask (the only case that imports nilearn);
    - a mask image and an ``(n, V)`` array: the array as it is.

    With a masker or a mask image the ``Domain`` is built from the mask
    (``masker.mask_img_``, or the mask image): a voxel is in the mask when its
    value is non-zero and finite, which is nilearn's rule. Node order is
    C-order over the mask, the column order of ``NiftiMasker.transform``
    (checked against nilearn 0.14.1 in
    ``tests/test_io.py::test_node_order_is_the_column_order_of_niftimasker_transform``).
    ``origin`` is the object given as ``space``.

    A masker is an object with ``transform`` and ``inverse_transform``; it is
    fitted when it also has ``mask_img_``. An image is an object with
    ``affine`` and ``dataobj``. Images as ``data`` are one image, one file
    name, or a list of them: whatever the masker's ``transform`` takes and
    turns into ``(n, V)``.

    ``conn``: 6, 18 or 26, the voxel connectivity of the ``Domain`` built
    from a masker or a mask image; ``None`` is 6. It must be left as ``None``
    with a ``Domain``.

    A ``MaskedData`` passed as ``data`` is returned unchanged; ``space`` and
    ``conn`` must then be left out.

    An array is 2-D, ``(n, V)`` with ``V`` the number of nodes. A
    C-contiguous float32 or float64 array is kept as it is, with no copy;
    anything else is converted to float64.

    Raises ``ValueError`` for images with no ``space`` or with a ``Domain``,
    a masker that is not fitted, a masker whose ``mask_img_`` is not an image
    (a surface masker), a ``space`` of another kind, a mask that is not 3-D,
    and data that are not ``(n, V)``.
    """
    if isinstance(data, MaskedData):
        if space is not None or conn is not None:
            raise ValueError("space and conn cannot be given with a MaskedData, which has its Domain already")
        return data
    images = _is_images(data)
    if isinstance(space, Domain):
        if conn is not None:
            raise ValueError("conn cannot be given with a Domain, which has its connectivity already")
        if images:
            raise ValueError("images need a fitted masker or a mask image as space, not a Domain")
        return MaskedData(_arrays.source(data, space.n_nodes), space, None)
    if space is None and images:
        raise ValueError("images need a space: a fitted masker or a mask image")
    mask_img = _mask_image(space, "space")
    if mask_img is None:
        raise ValueError(
            f"space must be a Domain, a fitted masker or a mask image, got {type(space).__name__}"
        )
    domain = Domain.from_mask(_arrays.image_mask(mask_img), 6 if conn is None else conn)
    if images:
        if _is_masker(space):
            data = space.transform(data)
        else:
            import copy

            from nilearn.maskers import NiftiMasker

            # nilearn zeroes the non-finite voxels of the mask it is given in place.
            data = NiftiMasker(mask_img=copy.deepcopy(space)).fit().transform(data)
    return MaskedData(_arrays.source(data, domain.n_nodes), domain, space)


def export_to(result, target=None) -> dict:
    """The maps of a ``TfceInference`` as images.

    Returns a dict of four images:

    - ``"stat"``: ``stat_obs``;
    - ``"tfce"``: ``tfce_obs``;
    - ``"logp_fwe"``: ``-log10(p_fwe)``;
    - ``"logp_unc"``: ``-log10(p_unc)``.

    Every image is 0 outside the mask. The p-maps are exported as
    ``-log10(p)``, the form of nilearn's ``logp_max_tfce``, so that 0 outside
    the mask reads as ``p = 1``. No p-value of a result is 0, so the maps are
    finite.

    ``target``: a fitted masker or a mask image. Left as ``None``, the maps go
    back through ``result.origin``. It is required when ``result.origin`` is
    ``None`` (a result computed from a ``Domain``, or read with
    ``TfceInference.load``), and it replaces the origin when given.

    Through a masker each map is ``masker.inverse_transform(map)`` (checked
    against nilearn 0.14.1 in
    ``tests/test_io.py::test_export_equals_inverse_transform``). Through a mask
    image each map is an image of the mask's class, built on the mask's affine
    and a copy of its header with the data type set to float64.

    Raises ``ValueError`` when ``result`` lacks a field of a
    ``TfceInference``, when there is neither an origin nor a target, when
    the target is not a fitted masker or a mask image, and when the mask does
    not have one voxel per node of the result.
    """
    if not all(hasattr(result, a) for a in ("origin", "stat_obs", "tfce_obs", "p_fwe", "p_unc")):
        raise ValueError(f"result must be a TfceInference, got {type(result).__name__}")
    host = result.origin if target is None else target
    if host is None:
        raise ValueError(
            "the result has no origin (it was computed from a Domain or loaded from a file); "
            "pass target, a fitted masker or a mask image"
        )
    mask_img = _mask_image(host, "target")
    if mask_img is None:
        raise ValueError(f"target must be a fitted masker or a mask image, got {type(host).__name__}")
    maps = {
        "stat": result.stat_obs,
        "tfce": result.tfce_obs,
        "logp_fwe": _arrays.neg_log10(result.p_fwe),
        "logp_unc": _arrays.neg_log10(result.p_unc),
    }
    domain = Domain.from_mask(_arrays.image_mask(mask_img))
    n_nodes = len(result.stat_obs)
    if domain.n_nodes != n_nodes:
        raise ValueError(f"the result has {n_nodes} nodes, the mask of target has {domain.n_nodes} voxels")
    if _is_masker(host):
        return {key: host.inverse_transform(vec) for key, vec in maps.items()}
    out = {}
    for key, vec in maps.items():
        volume = _arrays.scatter(vec, domain, 0.0)
        # The class copies the header, so the mask's own header stays as it is.
        img = type(mask_img)(volume, mask_img.affine, mask_img.header)
        # The header came with the mask's data type, often uint8; a file
        # written with it would round the map.
        img.set_data_dtype(volume.dtype)
        out[key] = img
    return out
