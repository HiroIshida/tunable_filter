## tunable filter 

Tunable filter is a library for tuning rgb-iamge filter (python 3.6 ~). By using this, one can
- easily compose your custom filter as shown in [`example/hsv_blur_crop.py`](example/hsv_blur_crop.py). 
- dump a simple yaml file corresponding to the tuning result for later use by `dump_yaml` method
- load the created yaml file to reconstruct the filter object by `from_yaml` classmethod.

### Installation
```
git clone git@github.com:HiroIshida/tunable_filter.git
cd tunable_filter
pip3 install -e .
```

A demo is in `example/hsv_blur_crop.py`.
### 1. You can first start tuning by
```
python3 example/hsv_blur_crop.py --tune
```

![demo](https://user-images.githubusercontent.com/38597814/163725554-b89412b9-b624-4cab-8906-b162827717a6.gif)

### 2. Then check that filter is reconstructed by
```
python3 example/hsv_blur_crop.py
```
