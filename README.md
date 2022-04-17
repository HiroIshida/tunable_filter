## tunable filter 
[![PyPI version](https://badge.fury.io/py/tunable-filter.svg)](https://pypi.org/project/tunable-filter/)

Tunable filter is a library for tuning rgb-iamge filter (python 3.6 ~). By using this, one can
- easily compose your custom filter as shown in [`tunable_filter/composite_zoo.py`](tunable_filter/composite_zoo.py). 
- dump a simple yaml file corresponding to the tuning result for later use by `dump_yaml` method
- load the created yaml file to reconstruct the filter object by `from_yaml` classmethod.

We are planning to add more filter primitives (e.g. RGB filter, morphological operation ...). 
However, user can easily define custom primitives inheriting one of   
- `LogicalFilterBase`, which outputs mask image with binary numpy array
- `FilterBase`, which converts image keeping the image size
- `ResizerBase`: which converts image changing the image sizes

and cane define custom composite filter by inheriting from `CompositeFilter`

### Installation
```
git clone git@github.com:HiroIshida/tunable_filter.git
cd tunable_filter
pip3 install -e .
```

A demo is in [`example/hsv_blur_crop.py`](example/hsv_blur_crop.py).
### 1. You can first start tuning by
```
python3 example/hsv_blur_crop.py --tune
```
![demo](https://user-images.githubusercontent.com/38597814/163730296-af5d78ab-43f2-479f-a196-4dcaf308ad1a.gif)

After this tuning, yaml file with the content below will be dumped. 
```
CropResizer:
  x_max: 357
  x_min: 25
  y_max: 457
  y_min: 74
GaussianBlurFilter:
  kernel_width: 9
HSVLogicalFilter:
  h_max: 69
  h_min: 31
  s_max: 228
  s_min: 56
  v_max: 255
  v_min: 0
ResolutionChangeResizer:
  resol: 136
```

### 2. Then check that filter is reconstructed by
```
python3 example/hsv_blur_crop.py
```
which load the yaml file by `from_yaml()` classmethod equiped with all `Tunable`. The output image will be as bellow. This shows that tuned filter is successfully reconstructd from the yaml file.

![debug](https://user-images.githubusercontent.com/38597814/163730440-155d88ad-fc45-47e6-b2e2-76dd639f5536.png)

