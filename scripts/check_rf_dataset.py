import numpy as np
from pathlib import Path

path = Path(
    r"data\xbd_s12\processed\palu_rf_dataset.npz"
)

with np.load(path, allow_pickle=False) as data:
    X = data["X"]
    y = data["y"]
    patch_ids = data["patch_ids"]
    feature_names = data["feature_names"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("patch_ids shape:", patch_ids.shape)
    print("Feature names:", len(feature_names))
    print("Unique labels:", np.unique(y))
    print("All features finite:", np.isfinite(X).all())
    print("Unique patches:", len(np.unique(patch_ids)))
    print("Feature names:", feature_names.tolist())

    assert X.shape == (522380, 24)
    assert len(X) == len(y) == len(patch_ids)
    assert len(feature_names) == 24
    assert np.isin(y, [0, 1]).all()
    assert np.isfinite(X).all()
    assert len(np.unique(y)) == 2

print("PASS: Dataset integrity checks succeeded.")