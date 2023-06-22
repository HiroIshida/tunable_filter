import argparse
import pprint
import cv2

from tunable_filter.composite_zoo import (
    HSVBlurCropResolFilter,
    GaussianBlurCropResolFilter,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="feedback mode")
    parser.add_argument("--gray", action="store_true", help="use gray image or not")
    args = parser.parse_args()
    tuning = args.tune
    gray = args.gray

    yaml_file_path = "/tmp/filter.yaml"
    if gray:
        img = cv2.imread("./media/dish.jpg", cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread("./media/dish.jpg")

    if tuning:
        if gray:
            tunable = GaussianBlurCropResolFilter.from_image(img)
        else:
            tunable = HSVBlurCropResolFilter.from_image(img)
        print("press q to finish tuning")
        tunable.launch_window()
        tunable.start_tuning(img)
        pprint.pprint(tunable.export_dict())
        tunable.dump_yaml(yaml_file_path)
    else:
        if gray:
            f = GaussianBlurCropResolFilter.from_yaml(yaml_file_path)
        else:
            f = HSVBlurCropResolFilter.from_yaml(yaml_file_path)
        img_filtered = f(img)
        cv2.imshow("debug", img_filtered)
        print("press q to terminate")
        while True:
            if cv2.waitKey(50) == ord("q"):
                break
