# Reference : https://github.com/yu4u/age-gender-estimation
import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from scipy.io import loadmat
from datetime import datetime
def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=32,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args

def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age

def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

def main():
    args = get_args()
    output_path = args.output
    db = args.db
    img_size = args.img_size
    min_score = args.min_score
    mat_path = "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    out_genders = []
    out_ages = []
    sample_num = len(face_score)
    out_imgs = np.empty((sample_num, img_size, img_size, 3), dtype=np.uint8)
    valid_sample_num = 0

    for i in tqdm(range(sample_num)):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        out_genders.append(int(gender[i]))
        out_ages.append(age[i])
        img = cv2.imread(str(full_path[i][0]))
        out_imgs[valid_sample_num] = cv2.resize(img, (img_size, img_size))
        valid_sample_num += 1

    output = {"image": out_imgs[:valid_sample_num], "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": db, "img_size": img_size, "min_score": min_score}
    scipy.io.savemat(output_path, output)


if __name__ == '__main__':
    main()
