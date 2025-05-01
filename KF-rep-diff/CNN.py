import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine


class KeyFrameAnalyzer:
    def __init__(self, video_path=None, image_dir=None):
        self.frames = []
        self.rgb_histograms = []
        self.hsv_histograms = []
        self.cnn_features = []
        self.video_info = {}

        if video_path:
            self.load_from_video(video_path)
        elif image_dir:
            self.load_from_directory(image_dir)

    def load_from_video(self, video_path, max_frames=100):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.video_info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }

        frame_count = self.video_info['frame_count']
        step = frame_count // max_frames if frame_count > max_frames else 1

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)
            count += 1
            if len(self.frames) >= max_frames:
                break

        cap.release()
        print(f"Loaded {len(self.frames)} frames from video")

    def load_from_directory(self, image_dir, max_frames=100):
        image_files = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))
        if not image_files:
            raise ValueError(f"No image files found in directory: {image_dir}")
        if len(image_files) > max_frames:
            step = len(image_files) // max_frames
            image_files = [image_files[i] for i in range(0, len(image_files), step)][:max_frames]
        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)
        print(f"Loaded {len(self.frames)} frames from directory")

    def compute_histograms(self, bins=32):
        self.rgb_histograms = []
        self.hsv_histograms = []
        for frame in self.frames:
            rgb_hist = []
            for i in range(3):
                hist = cv2.calcHist([frame], [i], None, [bins], [0, 256])
                hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                rgb_hist.extend(hist.flatten())
            self.rgb_histograms.append(np.array(rgb_hist))

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hsv_hist = []
            ranges = [180, 256, 256]
            for i in range(3):
                hist = cv2.calcHist([hsv_frame], [i], None, [bins], [0, ranges[i]])
                hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                hsv_hist.extend(hist.flatten())
            self.hsv_histograms.append(np.array(hsv_hist))

    def compute_frame_differences(self):
        if not self.rgb_histograms or not self.hsv_histograms:
            self.compute_histograms()

        rgb_diffs = []
        hsv_diffs = []
        for i in range(1, len(self.frames)):
            rgb_diff = 1 - cv2.compareHist(
                np.array(self.rgb_histograms[i - 1], dtype=np.float32),
                np.array(self.rgb_histograms[i], dtype=np.float32),
                cv2.HISTCMP_CORREL
            )
            hsv_diff = 1 - cv2.compareHist(
                np.array(self.hsv_histograms[i - 1], dtype=np.float32),
                np.array(self.hsv_histograms[i], dtype=np.float32),
                cv2.HISTCMP_CORREL
            )
            rgb_diffs.append(rgb_diff)
            hsv_diffs.append(hsv_diff)
        return {'rgb_differences': rgb_diffs, 'hsv_differences': hsv_diffs}

    def identify_key_frames(self, threshold=0.3):
        differences = self.compute_frame_differences()
        rgb_key_frames = [i + 1 for i, diff in enumerate(differences['rgb_differences']) if diff > threshold]
        hsv_key_frames = [i + 1 for i, diff in enumerate(differences['hsv_differences']) if diff > threshold]
        print(f"Identified {len(rgb_key_frames)} RGB key frames")
        print(f"Identified {len(hsv_key_frames)} HSV key frames")
        return {
            'rgb_key_frames': rgb_key_frames,
            'hsv_key_frames': hsv_key_frames,
            'differences': differences
        }

    def visualize_differences(self, threshold=0.3):
        differences = self.compute_frame_differences()
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(differences['rgb_differences'], 'r-', marker='o', label='RGB Difference')
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
        plt.title('RGB Frame Differences')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(differences['hsv_differences'], 'b-', marker='o', label='HSV Difference')
        plt.axhline(y=threshold, color='b', linestyle='--', alpha=0.5, label='Threshold')
        plt.title('HSV Frame Differences')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('frame_differences.png')
        plt.close()
        print("Frame differences visualization saved to 'frame_differences.png'")

    def display_key_frames(self, threshold=0.3):
        key_data = self.identify_key_frames(threshold)
        all_keys = sorted(set(key_data['rgb_key_frames'] + key_data['hsv_key_frames']))
        if not all_keys:
            print("No key frames identified. Try lowering the threshold.")
            return []
        n, cols = len(all_keys), min(4, len(all_keys))
        rows = (n + cols - 1) // cols
        plt.figure(figsize=(15, 5 * rows))
        for i, idx in enumerate(all_keys):
            plt.subplot(rows, cols, i + 1)
            title = f"Frame {idx}"
            if idx in key_data['rgb_key_frames']:
                title += " (RGB)"
            if idx in key_data['hsv_key_frames']:
                title += " (HSV)"
            plt.imshow(self.frames[idx])
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('key_frames.png')
        plt.close()
        print("Key frames visualization saved to 'key_frames.png'")
        return all_keys

    def extract_cnn_features(self):
        if not self.frames:
            raise ValueError("No frames loaded. Load frames first.")

        self.cnn_features = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(pretrained=True).to(device)
        model.eval()
        model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final layer

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        with torch.no_grad():
            for frame in self.frames:
                input_tensor = transform(frame).unsqueeze(0).to(device)
                features = model(input_tensor).squeeze().cpu().numpy()
                self.cnn_features.append(features)

        print(f"Extracted CNN features for {len(self.cnn_features)} frames.")

    def compute_cnn_differences(self):
        if not self.cnn_features:
            self.extract_cnn_features()
        cnn_diffs = []
        for i in range(1, len(self.cnn_features)):
            dist = cosine(self.cnn_features[i - 1], self.cnn_features[i])
            cnn_diffs.append(dist)
        return cnn_diffs

    def identify_cnn_key_frames(self, threshold=0.3):
        diffs = self.compute_cnn_differences()
        key_frames = [i + 1 for i, diff in enumerate(diffs) if diff > threshold]
        print(f"Identified {len(key_frames)} CNN key frames.")
        return key_frames

    def visualize_cnn_differences(self, threshold=0.3):
        diffs = self.compute_cnn_differences()
        plt.figure(figsize=(10, 4))
        plt.plot(diffs, 'g-', marker='o', label='CNN Difference')
        plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
        plt.title('CNN Frame Differences')
        plt.xlabel('Frame Index')
        plt.ylabel('Cosine Distance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cnn_frame_differences.png')
        plt.close()
        print("CNN frame differences visualization saved to 'cnn_frame_differences.png'")


if __name__ == "__main__":
    video_path = "vid2.mp4"

    analyzer = KeyFrameAnalyzer(video_path=video_path)

    print("\n--- Histogram-Based Analysis ---")
    analyzer.compute_histograms()
    analyzer.visualize_differences(threshold=0.3)
    analyzer.display_key_frames(threshold=0.3)

    print("\n--- CNN-Based Analysis ---")
    analyzer.extract_cnn_features()
    analyzer.visualize_cnn_differences(threshold=0.3)
    cnn_keys = analyzer.identify_cnn_key_frames(threshold=0.3)

    print("\nCNN Key Frame Indices:", cnn_keys)
    print("\nAnalysis complete!")
