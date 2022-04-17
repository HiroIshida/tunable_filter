import numpy as np
from tunable_filter.tunable import CompositeFilter
from tunable_filter.tunable import GaussianBlurFilter
from tunable_filter.tunable import HSVLogicalFilter
from tunable_filter.tunable import CropResizer
from tunable_filter.tunable import ResolutionChangeResizer


class HSVBlurCropResolFilter(CompositeFilter):
    # Define your custom filter by inheriting CompositeFilter
    # Only method you must implement is a factory method
    # In this example, the factory classmethod is named `from_image`
    # but whatever classmethod name is ok as long as it finally
    # calls construct_tunable to create the filter.
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
        resizers.append(ResolutionChangeResizer.default(resol_min=48, resol_max=248))
        return cls.construct_tunable(filteres, logical_filters, resizers)
