# 🐱🐶 Cat vs Dog CNN (Streamlit)

Upload a photo of a cat or dog and the CNN classifies it. A Grad-CAM heatmap shows where the model looked, and an optional energy-based filter rejects images that are neither.

## Layout
```
.
├── cats_vs_dogs_github_ready/
│   ├── app.py              <- Streamlit entry point
│   ├── requirements.txt    <- deployment dependencies
│   ├── catsdogs/           <- model + inference code (unchanged)
│   └── evaluate.py         <- needs scikit-learn (not required to run the app)
└── models/
    └── best_model.pt       <- trained weights (Git LFS)
```

## Run locally
```bash
cd cats_vs_dogs_github_ready
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io, click **Create app**, choose this repo.
3. Set **Main file path** to `cats_vs_dogs_github_ready/app.py`.
4. Deploy. Dependencies are read from `cats_vs_dogs_github_ready/requirements.txt`.

## Large model file (Git LFS)
`models/best_model.pt` is about 45 MB and tracked with Git LFS. The app detects when only the small LFS pointer was fetched and shows a clear error instead of crashing. If that happens on the deployed app, remove the LFS tracking so the real file is stored in git (45 MB is under GitHub's 100 MB limit):
```bash
git lfs untrack "*.pt"
git rm --cached models/best_model.pt
git add .gitattributes models/best_model.pt
git commit -m "Store model directly in git"
git push
```
