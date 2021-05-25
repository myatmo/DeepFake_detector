import os
from glob import glob
import argparse
import torch
from PIL import Image
from facenet_pytorch import MTCNN

# Determine if an nvidia GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


class FaceDetector:
    def __init__(self, image_size=(1920, 1080), margin=0, min_face_size=100):
        """
        Initializes the facial detector
        :param image_size: dimensions of imput images
        :param margin: margin to extend the captured image
        :param min_face_size: minimum detection size
        """
        self.image_size = image_size
        self.face_size = (160, 160)
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, min_face_size=min_face_size, thresholds=[0.99, 0.99, 0.99], selection_method='largest_over_theshold')

    @staticmethod
    def load_images(self, img_paths: list):
        return [self.load_image(img) for img in img_paths]

    def load_image(self, image_path: str):
        image = Image.open(image_path)
        if image.size != self.image_size:
            return None
        return image

    def generate_batch(self, img_paths: list):
        img_batch = {img: self.load_image(img) for img in img_paths}
        img_batch = {k: v for k, v in img_batch.items() if v is not None}
        return img_batch

    def detect(self, img_list: list):
        boxes, prob = self.mtcnn.detect(img_list)
        return boxes

    @staticmethod
    def crop(image, box):
        cropped = image.crop(box)
        return cropped


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Detect faces.')

    parser.add_argument('-i', '--input', metavar='INPUT DIRECTORY', type=str,
                        help='Source image directory for detecting faces')

    parser.add_argument('-o', '--output', metavar='OUTPUT DIRECTORY', type=str,
                        help='Output directory for storing face images')

    args = parser.parse_args()

    img_files = glob(f"{args.input}*.jpg")
    output_directory_path = args.output

    face_detector = FaceDetector()

    batch_size = 128

    # Split input image dataset into batches and feed into CNN
    for start_index in range(0, len(img_files), batch_size):
        # Collect batch of image file-paths
        batch = img_files[start_index:start_index + batch_size]

        # Generate image objects from paths
        image_data = face_detector.generate_batch(batch)

        image_paths = list(image_data.keys())
        images = list(image_data.values())

        # Feed image batch into network and get the bounding boxes (one per image)
        face_detections = face_detector.detect(images)

        # For each valid detection, crop the original image and save to disk
        for i in range(len(face_detections)):
            if face_detections[i] is not None:
                face_location = face_detections[i][0]
                face_image = face_detector.crop(images[i], face_location)
                new_file_name = output_directory_path + os.path.basename(image_paths[i])
                face_image.save(new_file_name)

        print(f"Batch completed: {start_index}")
