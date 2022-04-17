import argparse
import pprint
import cv2

from tunable_filter.composite_zoo import HSVBlurCropResolFilter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='feedback mode')
    args = parser.parse_args()
    tuning = args.tune

    yaml_file_path = '/tmp/filter.yaml'
    img = cv2.imread('./media/dish.jpg')

    if tuning:
        tunable = HSVBlurCropResolFilter.from_image(img)
        print('press q to finish tuning')
        tunable.start_tuning(img)
        pprint.pprint(tunable.export_dict())
        tunable.dump_yaml(yaml_file_path)
    else:
        f = HSVBlurCropResolFilter.from_yaml(yaml_file_path)
        img_filtered = f(img)
        cv2.imshow('debug', img_filtered)
        print('press q to terminate')
        while True:
            if cv2.waitKey(50) == ord('q'):
                break
