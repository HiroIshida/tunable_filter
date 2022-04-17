import numpy as np
from tunable_filter.tunable import CompositeFilter
from tunable_filter.tunable import GaussianBlurFilter
from tunable_filter.tunable import HSVLogicalFilter
from tunable_filter.tunable import CropLogicalFilter


class HSVBlurCropConverter(CompositeFilter):

    @classmethod
    def from_image(cls, img: np.ndarray):
        converters = []
        converters.append(GaussianBlurFilter.default())
        segmentors = []
        segmentors.append(HSVLogicalFilter.default())
        segmentors.append(CropLogicalFilter.from_image(img))
        return cls(converters, segmentors)


if __name__ == '__main__':
    img = np.random.randint(0, high=255, size=(224, 224, 3)).astype(np.uint8)
    tunable = HSVBlurCropConverter.from_image(img)
    tunable.start_tuning(img)
