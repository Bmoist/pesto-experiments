# Quick Guide for Pesto Model Training

This repo comes with MIR-1k downloaded inside data folder.

But in case you want to download the data by yourself, here's the tutorial:

## On Mac:
1. Create a folder named `data` in project root dir: `mkdir ./data`
2. Download MIR-1k from kaggle and put it inside this folder.
3. Generate csv file: 
   ```shell
   find MIR-1K/Wavfile -name "*.wav" | sort > mir-1k.csv
   find MIR-1K/PitchLabel -name "*.pv" | sort > mir-1k_annot.csv
   ```
4. Start training!
   ```shell
   python src/train.py data=mir-1k logger=csv
   ```
   - Optionally, on mac you may add `trainer=mps` or `trainer=cpu`

## On Windows
1. Create a folder named `data` in project root dir: `mkdir ./data`
2. Download MIR-1k from kaggle and put it inside this folder.
3. Create training data csv files:
   ```shell
   cd data
   Get-ChildItem -Recurse -Filter "Wavfile\*.wav" | Sort-Object -Property Name | Select-Object -ExpandProperty FullName > "..\mir-1k.csv"
   Get-ChildItem -Recurse -Filter "PitchLabel\*.pv" | Sort-Object -Property Name | Select-Object -ExpandProperty FullName > "..\mir-1k_annot.csv"   
   ```
4. Start training!
   ```shell
   python src/train.py data=mir-1k logger=csv
   ```