# DVC Quick Guide

## 1. Initialize DVC
```bash
git init
dvc init
```

## 2. Track a large data file
```bash
dvc add data/raw/large_file.csv
git add data/raw/large_file.csv.dvc .gitignore
git commit -m "Track dataset with DVC"
```

## 3. Configure a remote
```bash
dvc remote add -d storage s3://my-bucket/dvcstore
dvc remote modify storage endpointurl https://s3.example.com
```

## 4. Push data
```bash
dvc push
```

## 5. Pull data on another machine
```bash
git clone <repo-url>
dvc pull
```

## Tips
- Keep data out of Git, keep metadata in Git
- Use `dvc status` to see what changed
- Use `dvc repro` when you add pipelines
