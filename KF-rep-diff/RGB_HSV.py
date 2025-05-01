import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class KeyFrameAnalyzer:
    """
    A class for analyzing key frames in videos using RGB and HSV color spaces.
    """

    def __init__(self, video_path=None, image_dir=None):
        """
        Initialize the KeyFrameAnalyzer with either a video file or a directory of images.

        Args:
            video_path (str, optional): Path to the video file.
            image_dir (str, optional): Path to the directory containing image frames.
        """
        self.frames = []
        self.rgb_histograms = []
        self.hsv_histograms = []
        self.video_info = {}

        if video_path:
            self.load_from_video(video_path)
        elif image_dir:
            self.load_from_directory(image_dir)

    def load_from_video(self, video_path, max_frames=100):
        """
        Load frames from a video file.

        Args:
            video_path (str): Path to the video file.
            max_frames (int): Maximum number of frames to extract.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Store video information
        self.video_info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }

        frame_count = self.video_info['frame_count']
        if frame_count > max_frames:
            # Sample frames evenly
            step = frame_count // max_frames
        else:
            step = 1

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                self.frames.append(frame)

            count += 1

            if len(self.frames) >= max_frames:
                break

        cap.release()
        print(f"Loaded {len(self.frames)} frames from video")

    def load_from_directory(self, image_dir, max_frames=100):
        """
        Load frames from a directory of images.

        Args:
            image_dir (str): Path to the directory containing image frames.
            max_frames (int): Maximum number of frames to load.
        """
        image_files = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))

        if not image_files:
            raise ValueError(f"No image files found in directory: {image_dir}")

        # Sample frames evenly if there are more than max_frames
        if len(image_files) > max_frames:
            step = len(image_files) // max_frames
            image_files = [image_files[i] for i in range(0, len(image_files), step)][:max_frames]

        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                self.frames.append(frame)

        print(f"Loaded {len(self.frames)} frames from directory")

    def compute_histograms(self, bins=32):
        """
        Compute RGB and HSV histograms for all frames.

        Args:
            bins (int): Number of bins for the histograms.
        """
        if not self.frames:
            raise ValueError("No frames loaded. Load frames first.")

        # Clear existing histograms
        self.rgb_histograms = []
        self.hsv_histograms = []

        for frame in self.frames:
            # RGB histogram
            rgb_hist = []
            for i in range(3):  # R, G, B channels
                hist = cv2.calcHist([frame], [i], None, [bins], [0, 256])
                hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                rgb_hist.extend(hist.flatten())
            self.rgb_histograms.append(np.array(rgb_hist))

            # HSV histogram
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hsv_hist = []
            # H: 0-179, S: 0-255, V: 0-255 (OpenCV ranges)
            ranges = [180, 256, 256]
            for i in range(3):  # H, S, V channels
                hist = cv2.calcHist([hsv_frame], [i], None, [bins], [0, ranges[i]])
                hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                hsv_hist.extend(hist.flatten())
            self.hsv_histograms.append(np.array(hsv_hist))

    def compute_frame_differences(self):
        """
        Compute differences between consecutive frames using RGB and HSV histograms.

        Returns:
            dict: A dictionary containing RGB and HSV differences.
        """
        if not self.rgb_histograms or not self.hsv_histograms:
            self.compute_histograms()

        rgb_diffs = []
        hsv_diffs = []

        for i in range(1, len(self.frames)):
            # RGB difference (using correlation)
            rgb_diff = cv2.compareHist(
                np.array(self.rgb_histograms[i - 1], dtype=np.float32),
                np.array(self.rgb_histograms[i], dtype=np.float32),
                cv2.HISTCMP_CORREL
            )
            # Convert correlation to distance (1 - correlation)
            rgb_diffs.append(1 - rgb_diff)

            # HSV difference
            hsv_diff = cv2.compareHist(
                np.array(self.hsv_histograms[i - 1], dtype=np.float32),
                np.array(self.hsv_histograms[i], dtype=np.float32),
                cv2.HISTCMP_CORREL
            )
            hsv_diffs.append(1 - hsv_diff)

        return {
            'rgb_differences': rgb_diffs,
            'hsv_differences': hsv_diffs
        }

    def identify_key_frames(self, threshold=0.3):
        """
        Identify key frames based on the difference threshold.

        Args:
            threshold (float): Threshold for considering a frame as a key frame.

        Returns:
            dict: A dictionary containing RGB and HSV key frame indices.
        """
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
        """
        Visualize frame differences in both RGB and HSV spaces.

        Args:
            threshold (float): Threshold line to display on the plot.
        """
        differences = self.compute_frame_differences()

        plt.figure(figsize=(12, 8))

        # Plot RGB differences
        plt.subplot(2, 1, 1)
        plt.plot(differences['rgb_differences'], 'r-', marker='o', label='RGB Difference')
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
        plt.title('RGB Frame Differences')
        plt.xlabel('Frame Index')
        plt.ylabel('Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot HSV differences
        plt.subplot(2, 1, 2)
        plt.plot(differences['hsv_differences'], 'b-', marker='o', label='HSV Difference')
        plt.axhline(y=threshold, color='b', linestyle='--', alpha=0.5, label='Threshold')
        plt.title('HSV Frame Differences')
        plt.xlabel('Frame Index')
        plt.ylabel('Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('frame_differences.png')
        plt.close()  # Close the plot to free memory

        print("Frame differences visualization saved to 'frame_differences.png'")

    def display_key_frames(self, threshold=0.3):
        """
        Display only the key frames that exceed the threshold.

        Args:
            threshold (float): Threshold for key frame identification.

        Returns:
            list: Indices of all identified key frames.
        """
        key_frames_data = self.identify_key_frames(threshold)

        # Combine RGB and HSV key frames
        all_key_frames = sorted(set(key_frames_data['rgb_key_frames'] + key_frames_data['hsv_key_frames']))

        if not all_key_frames:
            print("No key frames identified. Try lowering the threshold.")
            return []

        # Calculate grid size
        n = len(all_key_frames)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))

        for i, frame_idx in enumerate(all_key_frames):
            plt.subplot(rows, cols, i + 1)

            # Mark frames detected by different methods
            title = f"Frame {frame_idx}"
            if frame_idx in key_frames_data['rgb_key_frames']:
                title += " (RGB)"
            if frame_idx in key_frames_data['hsv_key_frames']:
                title += " (HSV)"

            plt.imshow(self.frames[frame_idx])
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('key_frames.png')
        plt.close()

        print(f"Key frames visualization saved to 'key_frames.png'")

        return all_key_frames


if __name__ == "__main__":
    video_path = "vid2.mp4"  # Video file in the current directory

    # Create analyzer and run analysis
    analyzer = KeyFrameAnalyzer(video_path=video_path)
    analyzer.compute_histograms()

    # Visualize differences
    analyzer.visualize_differences()

    # Display key frames with different thresholds
    print("\nIdentifying key frames with threshold 0.3:")
    key_frames_0_3 = analyzer.display_key_frames(threshold=0.3)

    print("\nIdentifying key frames with threshold 0.2:")
    key_frames_0_2 = analyzer.display_key_frames(threshold=0.2)

    print("\nAnalysis complete!")