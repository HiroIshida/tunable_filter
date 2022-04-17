#!/usr/bin/env python3
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import queue
import yaml
import cv2
import numpy as np
from typing import Dict, List, Optional, Type, TypeVar

_window_name = 'window'
_initialized = {'?': False}


@dataclass
class TrackBarConfig:
    name: str
    val_min: int
    val_max: int


TunableT = TypeVar('TunableT', bound='Tunable')


@dataclass  # type: ignore
class Tunable(ABC):
    tunable: bool

    @abstractmethod
    def reflect_trackbar(self) -> None:
        pass

    @abstractmethod
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def export_dict(self) -> Dict:
        pass

    def dump_yaml(self, file_name: str) -> None:
        with open(file_name, 'w') as f:
            yaml.dump(self.export_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls: Type[TunableT], file_name: str) -> TunableT:
        with open(file_name, 'r') as f:
            dic = yaml.safe_load(f)
        return cls.from_dict(dic)

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[TunableT], dic: Dict) -> TunableT:
        pass

    def start_tuning(self, img: np.ndarray):
        assert self.tunable
        assert img.ndim == 3
        assert img.dtype == np.uint8
        while True:
            img_show = self.__call__(img)
            cv2.imshow(_window_name, img_show)
            self.reflect_trackbar()
            if cv2.waitKey(50) == ord('q'):
                break


@dataclass  # type: ignore
class TunablePrimitive(Tunable):
    configs: List[TrackBarConfig]
    values: Optional[Dict[str, int]] = None
    window_name: str = _window_name

    def __post_init__(self):
        if not self.tunable:
            return

        if not _initialized['?']:
            cv2.namedWindow(_window_name)
            _initialized['?'] = True
            print('initialize window')

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

    @classmethod
    def from_dict(cls, dic):
        return cls(False, [], dic)

    @classmethod
    def create_tunable(cls, configs: List[TrackBarConfig]):
        return cls(True, configs)

    def reflect_trackbar(self):
        for config in self.configs:
            self.values[config.name] = cv2.getTrackbarPos(config.name, self.window_name)

    def export_dict(self) -> Dict:
        assert self.values is not None
        return self.values


class LogicalFilterBase(TunablePrimitive):

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        assert rgb.ndim == 3
        assert rgb.dtype == np.uint8
        out = self._call_impl(rgb)
        assert rgb.shape[:2] == out.shape[:2]
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
        assert rgb.shape == out.shape
        assert out.ndim == 3
        assert out.dtype == np.uint8
        return out

    @abstractmethod
    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        pass


class ResizerBase(TunablePrimitive):

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        assert rgb.ndim == 3
        assert rgb.dtype == np.uint8
        out = self._call_impl(rgb)
        if out.shape[0] < 5 or out.shape[1] < 5:
            out = np.zeros((5, 5, 3), dtype=np.uint8)
        assert out.ndim == 3
        assert out.dtype == np.uint8
        return out

    @abstractmethod
    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        pass


class CropLogicalFilter(LogicalFilterBase):

    @classmethod
    def from_image(cls, rgb: np.ndarray):
        width, height, _ = rgb.shape
        configs = []
        configs.append(TrackBarConfig('crop_x_min', 0, width))
        configs.append(TrackBarConfig('crop_x_max', 0, width))
        configs.append(TrackBarConfig('crop_y_min', 0, height))
        configs.append(TrackBarConfig('crop_y_max', 0, height))
        return cls.create_tunable(configs)

    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        assert self.values is not None
        arr = np.ones(rgb.shape[:2], dtype=bool)
        arr[:self.values['crop_x_min'], :] = False
        arr[self.values['crop_x_max']:, :] = False
        arr[:, :self.values['crop_y_min']] = False
        arr[:, self.values['crop_y_max']:] = False
        return arr


class HSVLogicalFilter(LogicalFilterBase):

    @classmethod
    def default(cls):
        configs = []
        configs.append(TrackBarConfig('h_min', 0, 255))
        configs.append(TrackBarConfig('h_max', 0, 255))
        configs.append(TrackBarConfig('s_min', 0, 255))
        configs.append(TrackBarConfig('s_max', 0, 255))
        configs.append(TrackBarConfig('v_min', 0, 255))
        configs.append(TrackBarConfig('v_max', 0, 255))
        return cls.create_tunable(configs)

    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        assert self.values is not None
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        bool_mat = np.ones(rgb.shape[:2], dtype=bool)
        for i, t in enumerate(['h', 's', 'v']):
            key_min = t + '_min'
            key_max = t + '_max'
            b_min = self.values[key_min]
            b_max = self.values[key_max]
            bool_mat_local = np.logical_and(hsv[:, :, i] >= b_min, hsv[:, :, i] <= b_max)
            bool_mat *= bool_mat_local
        return bool_mat


class GaussianBlurFilter(FilterBase):

    @classmethod
    def default(cls):
        configs = []
        configs.append(TrackBarConfig('kernel_width', 1, 20))
        return cls.create_tunable(configs)

    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        assert self.values is not None
        width = self.values['kernel_width']
        blured = cv2.blur(rgb, (width, width))
        return blured


class CropResizer(ResizerBase):

    @classmethod
    def from_image(cls, rgb: np.ndarray):
        width, height, _ = rgb.shape
        configs = []
        configs.append(TrackBarConfig('crop_x_min', 0, width))
        configs.append(TrackBarConfig('crop_x_max', 0, width))
        configs.append(TrackBarConfig('crop_y_min', 0, height))
        configs.append(TrackBarConfig('crop_y_max', 0, height))
        return cls.create_tunable(configs)

    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        assert self.values is not None
        out = rgb[self.values['crop_x_min']:self.values['crop_x_max'], self.values['crop_y_min']:self.values['crop_y_max']]
        return out


def get_all_concrete_tunable_primitive_types() -> List[Type[TunablePrimitive]]:
    concrete_types: List[Type] = []
    q = queue.Queue()  # type: ignore
    q.put(TunablePrimitive)
    while not q.empty():
        t: Type = q.get()
        if len(t.__subclasses__()) == 0:
            concrete_types.append(t)

        for st in t.__subclasses__():
            q.put(st)
    return list(set(concrete_types))


@dataclass
class CompositeFilter(Tunable):
    filters: List[FilterBase]
    logical_filters: List[LogicalFilterBase]
    resizers: List[ResizerBase]

    def __call__(self, img_inp: np.ndarray) -> np.ndarray:
        img_out = deepcopy(img_inp)
        for converter in self.filters:
            img_out = converter(img_out)
        bool_mat = np.ones(img_out.shape[:2], dtype=bool)
        for segmentor in self.logical_filters:
            bool_mat *= segmentor(img_out)
        img_out[np.logical_not(bool_mat)] = (0, 0, 0)

        for resizer in self.resizers:
            img_out = resizer(img_out)
        return img_out

    def reflect_trackbar(self) -> None:
        for primitives in [self.filters, self.logical_filters, self.resizers]:
            for p in primitives:  # type: ignore
                p.reflect_trackbar()

    @classmethod
    def construct_untunable(cls,
                            filters: List[FilterBase],
                            logical_filters: List[LogicalFilterBase],
                            resizers: List[ResizerBase]) -> 'CompositeFilter':
        return cls(False, filters, logical_filters, resizers)

    @classmethod
    def from_dict(cls, dic) -> 'CompositeFilter':
        types = get_all_concrete_tunable_primitive_types()

        filters = []
        logical_filters = []
        resizers = []
        for key, subdict in dic.items():
            for t in types:
                if key == t.__name__:
                    primitive = t.from_dict(subdict)
                    if issubclass(t, FilterBase):
                        filters.append(primitive)
                    if issubclass(t, LogicalFilterBase):
                        logical_filters.append(primitive)
                    if issubclass(t, ResizerBase):
                        resizers.append(primitive)

        return cls.construct_untunable(filters, logical_filters, resizers)

    def export_dict(self) -> Dict:
        d = {}
        for primitives in [self.filters, self.logical_filters, self.resizers]:
            for p in primitives:  # type: ignore
                d[p.__class__.__name__] = p.values
        return d
