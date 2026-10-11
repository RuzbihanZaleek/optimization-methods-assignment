# Data

## Source

"Customer Information" by batoulmahvash on Kaggle:
https://www.kaggle.com/datasets/batoulmahvash/customer-information/data

The Kaggle licence is listed as "Unknown", so the file is **not** stored in this repository.

## Download (Windows PowerShell, from the repository root)

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
Invoke-WebRequest -Uri "https://www.kaggle.com/api/v1/datasets/download/batoulmahvash/customer-information" -OutFile data\raw\customer-information.zip
Expand-Archive -Path data\raw\customer-information.zip -DestinationPath data\raw -Force
Remove-Item data\raw\customer-information.zip
```

If the command fails, download the ZIP manually from the Kaggle page (free account needed)
and put `Customer Data.xlsx` in `data/raw/`.

Expected file: `data/raw/Customer Data.xlsx` (169,521 bytes,
SHA-256 `5DFF5933F68EAEB4BAB743D440E0FF2B159EBBC8177F2867DC79890225DB23D7`).

## Processed data

`python -m src.data_prep` creates `data/processed/customers_clean.csv` (87 customers).
This folder is also ignored by Git because it is generated.
