#!/usr/bin/env python3
from abc import ABC, abstractmethod
from copy import deepcopy
import argparse
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Dict, List, Optional

@dataclass
class TrackBarConfig:
    name: str
    val_min: int
    val_max: int


class Tunable(ABC):

    @abstractmethod
    def reflect_trackbar(self) -> None:
        pass

    @abstractmethod
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        pass


@dataclass
class TunablePrimitive(Tunable):
    configs: List[TrackBarConfig]
    values: Optional[Dict[str, int]] = None
    window_name: str = 'window'

    def __post_init__(self):
        if self.values is None:
            # auto set initial values
            self.values = {}
            for config in self.configs:
                cv2.createTrackbar(
                        config.name,
                        self.window_name,
                        config.val_min,
                        config.val_max,
                        lambda x: None)
                self.values[config.name] = int(0.5 * (config.val_min + config.val_max))

    def reflect_trackbar(self):
        for config in self.configs:
            self.values[config.name] = cv2.getTrackbarPos(config.name, self.window_name)


class LogicalFilterBase(TunablePrimitive):

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        assert rgb.ndim == 3
        assert rgb.dtype == np.uint8
        out = self._call_impl(rgb)
        assert out.ndim == 2
        assert out.dtype == bool
        return out

    @abstractmethod
    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        pass


class FilterBase(TunablePrimitive):

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        assert rgb.ndim == 3
        assert rgb.dtype == np.uint8
        out = self._call_impl(rgb)
        assert out.ndim == 3
        assert out.dtype == np.uint8
        return out

    @abstractmethod
    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        pass


@dataclass
class CropLogicalFilter(LogicalFilterBase):

    @classmethod
    def from_image(cls, rgb: np.ndarray):
        width, height, _ = rgb.shape
        configs = []
        configs.append(TrackBarConfig('crop_x_min', 0, width))
        configs.append(TrackBarConfig('crop_x_max', 0, width))
        configs.append(TrackBarConfig('crop_y_min', 0, height))
        configs.append(TrackBarConfig('crop_y_max', 0, height))
        return cls(configs)

    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        assert self.values is not None
        arr = np.ones(rgb.shape[:2], dtype=bool)
        arr[:self.values['crop_x_min'], :] = False
        arr[self.values['crop_x_max']:, :] = False
        arr[:, :self.values['crop_y_min']] = False
        arr[:, self.values['crop_y_max']:] = False
        return arr

@dataclass
class GaussianBlurFilter(FilterBase):

    @classmethod
    def default(cls):
        configs = []
        configs.append(TrackBarConfig('kernel_width', 1, 20))
        return cls(configs)

    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        assert self.values is not None
        width = self.values['kernel_width']
        blured = cv2.blur(rgb, (width, width))
        return blured


@dataclass
class CompositeFilter(Tunable):
    converters: List[FilterBase]
    segmetors: List[LogicalFilterBase]

    def __call__(self, img_inp: np.ndarray) -> np.ndarray:
        img_out = deepcopy(img_inp)
        for converter in self.converters:
            img_out = converter(img_out)
        bool_mat = np.ones(img_out.shape[:2], dtype=bool)
        for segmentor in self.segmetors:
            bool_mat *= segmentor(img_out)
        img_out[np.logical_not(bool_mat)] = (0, 0, 0)
        return img_out

    def reflect_trackbar(self) -> None:
        for primitive in self.converters + self.segmetors:
            primitive.reflect_trackbar()


class BlurCropConverter(CompositeFilter):

    @classmethod
    def from_image(cls, img: np.ndarray):
        converters = []
        converters.append(GaussianBlurFilter.default())
        segmentors = []
        segmentors.append(CropLogicalFilter.from_image(img))
        return cls(converters, segmentors)


if __name__ == '__main__':
    img = np.random.randint(0, high=255, size=(224, 224, 3)).astype(np.uint8)
    cv2.namedWindow('window')

    composite_converter = BlurCropConverter.from_image(img)

    while True:
        img_show = composite_converter(img)
        cv2.imshow('window', img_show)
        composite_converter.reflect_trackbar()

        if cv2.waitKey(50) == ord('q'):
            break
