import argparse
import pprint
import cv2

from tunable_filter.composite_zoo import HSVBlurCropResolFilter, BlurCropResolFilter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='feedback mode')
    parser.add_argument('--gray', action='store_true', help='use gray scale image or not')
    args = parser.parse_args()
    tuning = args.tune
    gray = args.gray

    yaml_file_path = '/tmp/filter.yaml'
    img = cv2.imread('./media/dish.jpg', cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    filter_type = BlurCropResolFilter if gray else HSVBlurCropResolFilter

    if tuning:
        tunable = filter_type.from_image(img)
        print('press q to finish tuning')
        tunable.launch_window()
        tunable.start_tuning(img)
        pprint.pprint(tunable.export_dict())
        tunable.dump_yaml(yaml_file_path)
    else:
        f = filter_type.from_yaml(yaml_file_path)
        img_filtered = f(img)
        cv2.imshow('debug', img_filtered)
        print('press q to terminate')
        while True:
            if cv2.waitKey(50) == ord('q'):
                break
