import argparse
import pprint
import cv2
import numpy as np
from tunable_filter.tunable import CompositeFilter
from tunable_filter.tunable import GaussianBlurFilter
from tunable_filter.tunable import HSVLogicalFilter
from tunable_filter.tunable import CropResizer


class HSVBlurCropFilter(CompositeFilter):
    # Define your custom filter by inheriting CompositeFilter
    # Only method you must implement is a factory method
    # In this example, the factory classmethod is named `from_image`
    # but whatever classmethod name is ok as long as it finally
    # calls construct_untunable to create the filter.
    # As for factory classmethod in python, prease see:
    # https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-implement-multiple-constructors/682545#682545

    @classmethod
    def from_image(cls, img: np.ndarray):
        filteres = []
        filteres.append(GaussianBlurFilter.default())
        logical_filters = []
        logical_filters.append(HSVLogicalFilter.default())
        resizers = []
        resizers.append(CropResizer.from_image(img))
        return cls.construct_untunable(filteres, logical_filters, resizers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='feedback mode')
    args = parser.parse_args()
    tuning = args.tune

    yaml_file_path = '/tmp/filter.yaml'
    img = cv2.imread('./media/dish.jpg')

    if tuning:
        tunable = HSVBlurCropFilter.from_image(img)
        print('press q to finish tuning')
        tunable.start_tuning(img)
        pprint.pprint(tunable.export_dict())
        tunable.dump_yaml(yaml_file_path)
    else:
        f = HSVBlurCropFilter.from_yaml(yaml_file_path)
        img_filtered = f(img)
        cv2.imshow('debug', img_filtered)
        print('press q to terminate')
        while True:
            if cv2.waitKey(50) == ord('q'):
                break
