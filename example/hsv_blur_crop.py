import pprint
import cv2
import numpy as np
from tunable_filter.tunable import CompositeFilter
from tunable_filter.tunable import GaussianBlurFilter
from tunable_filter.tunable import HSVLogicalFilter
from tunable_filter.tunable import CropResizer


class HSVBlurCropConverter(CompositeFilter):

    @classmethod
    def from_image(cls, img: np.ndarray):
        converters = []
        converters.append(GaussianBlurFilter.default())
        segmentors = []
        segmentors.append(HSVLogicalFilter.default())
        resizers = []
        resizers.append(CropResizer.from_image(img))
        return cls(True, converters, segmentors, resizers)


if __name__ == '__main__':
    img = cv2.imread('./dish.jpg')
    tunable = HSVBlurCropConverter.from_image(img)
    print('press q to finish tuning')
    tunable.start_tuning(img)
    dic = tunable.export_dict()
    pprint.pprint(dic)
