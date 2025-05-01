import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class KeyFrameAnalyzerSIFT:
    """
    A class for analyzing key frames in videos using SIFT features.
    """

    def __init__(self, video_path=None, image_dir=None):
        self.frames = []
        self.gray_frames = []  # Store grayscale versions for SIFT
        self.sift = cv2.SIFT_create()
        self.video_info = {}

        if video_path:
            self.load_from_video(video_path)
        elif image_dir:
            self.load_from_directory(image_dir)

    def load_from_video(self, video_path, max_frames=100):
        """Same as before, but also store grayscale frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.video_info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }

        frame_count = self.video_info['frame_count']
        step = max(1, frame_count // max_frames)

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frames.append(rgb_frame)
                self.gray_frames.append(gray_frame)

            count += 1
            if len(self.frames) >= max_frames:
                break

        cap.release()
        print(f"Loaded {len(self.frames)} frames from video")

    def load_from_directory(self, image_dir, max_frames=100):
        """Same as before, but with grayscale conversion."""
        image_files = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))
        if not image_files:
            raise ValueError(f"No image files found in directory: {image_dir}")

        if len(image_files) > max_frames:
            step = len(image_files) // max_frames
            image_files = [image_files[i] for i in range(0, len(image_files), step)][:max_frames]

        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frames.append(rgb_frame)
                self.gray_frames.append(gray_frame)

        print(f"Loaded {len(self.frames)} frames from directory")

    def compute_sift_descriptors(self):
        """Compute SIFT descriptors for all frames."""
        self.descriptors = []
        for gray_frame in self.gray_frames:
            _, desc = self.sift.detectAndCompute(gray_frame, None)
            if desc is not None:
                self.descriptors.append(desc)
            else:
                # If no features found, use empty array
                self.descriptors.append(np.array([]))

    def compute_frame_differences(self):
        """Compute differences between consecutive frames using SIFT feature matching."""
        if not hasattr(self, 'descriptors'):
            self.compute_sift_descriptors()

        differences = []
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        for i in range(1, len(self.descriptors)):
            desc1 = self.descriptors[i - 1]
            desc2 = self.descriptors[i]

            if len(desc1) == 0 or len(desc2) == 0:
                # If no descriptors, assume maximum difference
                differences.append(1.0)
                continue

            matches = bf.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Normalized average distance (0=identical, 1=completely different)
            avg_distance = np.mean([m.distance for m in matches])
            max_possible_distance = 300  # Empirical value for SIFT
            normalized_diff = min(avg_distance / max_possible_distance, 1.0)
            differences.append(normalized_diff)

        return differences

    def identify_key_frames(self, threshold=0.3):
        """Identify key frames where SIFT difference exceeds threshold."""
        differences = self.compute_frame_differences()
        key_frames = [i + 1 for i, diff in enumerate(differences) if diff > threshold]
        print(f"Identified {len(key_frames)} key frames using SIFT")
        return {
            'key_frames': key_frames,
            'differences': differences
        }

    def visualize_differences(self, threshold=0.3):
        """Visualize SIFT frame differences."""
        result = self.identify_key_frames(threshold)
        differences = result['differences']

        plt.figure(figsize=(12, 4))
        plt.plot(differences, 'g-', marker='o', label='SIFT Difference')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.title('SIFT Frame Differences')
        plt.xlabel('Frame Index')
        plt.ylabel('Normalized Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('sift_differences.png')
        plt.close()
        print("SIFT differences visualization saved to 'sift_differences.png'")

    def display_key_frames(self, threshold=0.3):
        """Display identified key frames."""
        result = self.identify_key_frames(threshold)
        key_frames = result['key_frames']

        if not key_frames:
            print("No key frames identified. Try lowering the threshold.")
            return []

        n = len(key_frames)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))
        for i, frame_idx in enumerate(key_frames):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(self.frames[frame_idx])
            plt.title(f"Frame {frame_idx}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('sift_key_frames.png')
        plt.close()
        print(f"SIFT key frames visualization saved to 'sift_key_frames.png'")
        return key_frames


if __name__ == "__main__":
    video_path = "vid2.mp4"  # Change to your video path

    # SIFT-based analysis
    analyzer = KeyFrameAnalyzerSIFT(video_path=video_path)

    print("\nIdentifying key frames with SIFT (threshold=0.3):")
    analyzer.visualize_differences(threshold=0.3)
    key_frames = analyzer.display_key_frames(threshold=0.3)

    print("\nAnalysis complete!")