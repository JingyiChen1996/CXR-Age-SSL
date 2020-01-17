# CXR-Age-SSL
Evaluating different semi-supervised learning algorithms for CXR age estimation

### Data
NIH Dataset (To fasten training, I only used 10,000 images for training. Please see 'data' folder for more information)

### Supervised learning
```
python3 supervised-train.py --gpu=0 --out='result/supervised' 
```

### Semi-supervised learning
#### MixMatch
```
python3 ssl-train.py --gpu=1 --workers=16 --out='result/mixmatch' --model='mixmatch' --n-labeled=1000
```
#### PI Model
```
python3 ssl-train.py --gpu=2 --workers=16 --out='result/pi' --model='pi' -n-labeled=1000
```

#### Mean Teacher
```
python3 ssl-train.py --gpu=3 --workers=16 --out='reslut/mt' --model='mt' --n-labeled=1000
```

### Grad Cam
```
python3 grad-cam.py
```
