# data/

## Dataset
**New Plant Diseases Dataset** (Kaggle, `vipoooool/new-plant-diseases-dataset`) — 87,000+ RGB leaf images,
38 classes, an 80/20 train/validation split (offline-augmented version of PlantVillage) and a folder of 33 loose test photos.
Link: <https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset>

We use only **Apple** and **Corn (maize)** — 8 classes:

| Crop | Classes |
|---|---|
| Apple | Apple scab · Black rot · Cedar apple rust · Healthy |
| Corn | Cercospora leaf spot / Gray leaf spot · Common rust · Northern leaf blight · Healthy |

Subset used for all experiments (random seed 42): **500 training + 100 validation images per class**
(4,000 train / 800 validation), plus the **10 Apple/Corn test photos** from the dataset's test folder.
Only these 10 test images exist in the dataset for our crops, so we also tried the models on part of the training set;
real-world accuracy still has to be measured on a farm.

## Folders (created by the scripts — not stored in git)
| Folder | Created by | Content |
|---|---|---|
| `raw` / Kaggle cache | `preprocessing.download_dataset` | the full dataset (location saved in `dataset_location.txt`) |
| `subset/train`, `subset/valid` | `preprocessing.create_subset` | the sampled Apple + Corn images |
| `test_images/` | `preprocessing.create_subset` | the 10 test photos — **commit these** (tiny) |
| `augmented/svm/train`, `augmented/cnn/train` | `preprocessing.augment` | originals + ×3 / ×2 augmented copies |
| `features/` | `vgg_svm.train` | cached VGG16 feature vectors (`.npy`) |

## Why is the dataset not inside the repository?
The full dataset is ~2 GB and GitHub rejects files above 100 MB. The scripts rebuild exactly the same subset from Kaggle
(fixed seed). If your course needs the subset itself in the repo, zip `data/subset` and `data/test_images`
(≈ 100 MB for 4,800 images) and either use Git LFS or put the zip on Google Drive and paste the link here.
