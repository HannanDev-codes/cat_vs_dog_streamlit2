# Dataset

Unzip your `archive_3.zip` **into this folder** so it looks like this:

```
data/
├── train/
│   ├── cat/   (95 images)
│   └── dog/   (180 images)
└── val/
    ├── cat/   (24 images)
    └── dog/   (46 images)
```

The folder name is the label. The `train.csv`, `val.csv` and `classname.txt` files that come in the zip
are not needed by this code, so you can leave them or delete them.

Dataset images are not committed to GitHub (see `.gitignore`) because they are not ours to redistribute.
If you want them in your repo anyway, remove the `data/*` lines from `.gitignore`.
