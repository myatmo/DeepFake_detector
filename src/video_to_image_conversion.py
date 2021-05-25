import argparse
import os
import json
import cv2
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert videos to images.')

    parser.add_argument('-i', '--input', metavar='INPUT DIRECTORY', type=str,
                        help='Wildcard path for video files')

    parser.add_argument('-m', '--metadata', metavar='METADATA PATH', type=str,
                        help='Wildcard path for metadata.json files')

    parser.add_argument('-o', '--output', metavar='OUTPUT DIRECTORY', type=str,
                        help='Output directory for storing sampled images')

    parser.add_argument('-mo', '--meta_output', metavar='METADATA OUTPUT FILE', type=str,
                        help='Output path for storing combined metadata.json')

    parser.add_argument('-s', '--samples', metavar='SAMPLING FREQUENCY', type=int,
                        help='Sampling frequency for extracting frames from video (e.g. 60 -> Get image every 60 frames')

    args = parser.parse_args()

    video_filenames = glob(args.input)
    num_examples = len(video_filenames)
    subset_vid_names = video_filenames[:num_examples]
    meta_files = glob(args.metadata)
    combined_meta_file = args.meta_output

    sample_frames = list(range(0, 300, int(args.samples)))

    metadata = {}
    for meta_file in meta_files:
        with open(meta_file) as f:
            data = json.load(f)
        metadata.update(data)

    labels = {}

    for vid in subset_vid_names:
        vid_name = os.path.basename(vid)
        label = metadata[vid_name]['label']
        for sample in sample_frames:
            name = vid_name.split(".")[0] + "_" + str(sample) + ".jpg"
            labels[name] = label

    with open(combined_meta_file, "w+") as fp:
        json.dump(labels, fp)

    count = 0
    last_name = ""
    for vid in subset_vid_names:
        img_name = os.path.basename(vid).split(".")[0]
        cap = cv2.VideoCapture(vid)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if i in sample_frames:
                name = "/home/madhu/data/preprocessed/dfdc/" + img_name + "_" + str(i) + ".jpg"
                cv2.imwrite(name, frame)
                last_name = name
            i += 1
        count += 1
        cap.release()
        if count == 1:
            print(last_name)
        if count % 50 == 0:
            print("Videos completed: " + str(count))
