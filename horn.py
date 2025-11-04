import cv2
import numpy as np
import librosa
import json
import os
from datetime import datetime, timedelta
from scipy import signal
import argparse
import logging
import matplotlib.pyplot as plt

class RailwayHornDetector:
    def __init__(self, config=None):
        """
        Initialize the Railway Horn Detector
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {
            'horn_freq_range': (500, 2000),  # Hz - Wider range for analysis
            'horn_peak_freq': (500, 600),   # Hz - Wider peak range
            'min_horn_duration': 1,        # seconds - Shorter minimum
            'amplitude_threshold': 0.001,    # Much lower threshold
            'analysis_mode': True,           # Enable detailed analysis
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def extract_audio_from_video(self, video_path):
        """Extract audio from MP4 video file"""
        try:
            # Load audio using librosa (handles MP4 containers)
            audio, sr = librosa.load(video_path, sr=None)
            self.logger.info(f"Audio extracted: {len(audio)} samples at {sr} Hz")
            return audio, sr
        except Exception as e:
            self.logger.error(f"Error extracting audio: {e}")
            return None, None
    
    def analyze_audio_characteristics(self, audio, sr, output_dir="analysis"):
        """Analyze audio to understand its characteristics and help tune parameters"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("Analyzing audio characteristics...")
        
        # 1. Overall audio statistics
        audio_stats = {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'max_amplitude': float(np.max(np.abs(audio))),
            'mean_amplitude': float(np.mean(np.abs(audio))),
            'rms_amplitude': float(np.sqrt(np.mean(audio**2)))
        }
        
        # 2. Frequency analysis using FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        magnitude = np.abs(fft)
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(magnitude, height=np.max(magnitude)*0.1)[0]
        dominant_freqs = freqs[peak_indices]
        dominant_mags = magnitude[peak_indices]
        
        # Sort by magnitude
        sorted_indices = np.argsort(dominant_mags)[::-1]
        top_frequencies = [(float(dominant_freqs[i]), float(dominant_mags[i])) 
                          for i in sorted_indices[:10]]
        
        # 3. Short-time analysis
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = int(0.02 * sr)   # 20ms hop for better resolution
        
        # Calculate various features
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # 4. Find potential horn sections using multiple criteria
        # High energy sections
        energy_threshold = np.percentile(rms, 85)  # Top 15% energy
        high_energy_mask = rms > energy_threshold
        
        # Spectral centroid analysis (horns often have specific spectral characteristics)
        centroid_mean = np.mean(spectral_centroids)
        centroid_std = np.std(spectral_centroids)
        interesting_centroid_mask = (spectral_centroids > centroid_mean + 0.5 * centroid_std) | \
                                   (spectral_centroids < centroid_mean - 0.5 * centroid_std)
        
        # Combine criteria
        potential_horn_mask = high_energy_mask & interesting_centroid_mask
        
        # Find continuous segments
        potential_segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_potential in enumerate(potential_horn_mask):
            if is_potential and not in_segment:
                segment_start = i
                in_segment = True
            elif not is_potential and in_segment:
                segment_end = i
                duration = (segment_end - segment_start) * hop_length / sr
                if duration >= 0.2:  # At least 200ms
                    potential_segments.append({
                        'start_time': times[segment_start],
                        'end_time': times[segment_end],
                        'duration': duration,
                        'avg_rms': np.mean(rms[segment_start:segment_end]),
                        'avg_centroid': np.mean(spectral_centroids[segment_start:segment_end])
                    })
                in_segment = False
        
        # Handle last segment
        if in_segment:
            segment_end = len(potential_horn_mask)
            duration = (segment_end - segment_start) * hop_length / sr
            if duration >= 0.2:
                potential_segments.append({
                    'start_time': times[segment_start],
                    'end_time': times[segment_end],
                    'duration': duration,
                    'avg_rms': np.mean(rms[segment_start:segment_end]),
                    'avg_centroid': np.mean(spectral_centroids[segment_start:segment_end])
                })
        
        # 5. Create visualizations
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Waveform
        plt.subplot(4, 1, 1)
        time_axis = np.linspace(0, len(audio)/sr, len(audio))
        plt.plot(time_axis[:len(audio)//100], audio[:len(audio)//100])  # Plot first 1% for visibility
        plt.title('Audio Waveform (First 1% of file)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot 2: Frequency spectrum
        plt.subplot(4, 1, 2)
        plt.plot(freqs[:len(freqs)//10], magnitude[:len(magnitude)//10])  # Plot up to 1/10 of frequencies
        plt.title('Frequency Spectrum (Low Frequencies)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(500, 2000)  # Focus on relevant frequency range
        
        # Plot 3: RMS Energy over time
        plt.subplot(4, 1, 3)
        plt.plot(times, rms)
        plt.axhline(y=energy_threshold, color='r', linestyle='--', label=f'Energy Threshold ({energy_threshold:.4f})')
        for segment in potential_segments:
            plt.axvspan(segment['start_time'], segment['end_time'], alpha=0.3, color='yellow')
        plt.title('RMS Energy Over Time (Yellow = Potential Horn Segments)')
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')
        plt.legend()
        
        # Plot 4: Spectral Centroid
        plt.subplot(4, 1, 4)
        plt.plot(times, spectral_centroids)
        plt.axhline(y=centroid_mean, color='g', linestyle='--', label=f'Mean Centroid ({centroid_mean:.0f} Hz)')
        plt.title('Spectral Centroid Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'audio_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Convert NumPy types to native Python types in top_frequencies
        top_frequencies = [(float(freq), float(mag)) 
                          for freq, mag in top_frequencies]
        
        # Convert NumPy types in potential_segments
        for segment in potential_segments:
            segment['start_time'] = float(segment['start_time'])
            segment['end_time'] = float(segment['end_time'])
            segment['duration'] = float(segment['duration'])
            segment['avg_rms'] = float(segment['avg_rms'])
            segment['avg_centroid'] = float(segment['avg_centroid'])
        
        # 6. Generate analysis report with converted types
        analysis_report = {
            'audio_statistics': {
                'duration': float(len(audio) / sr),
                'sample_rate': int(sr),
                'max_amplitude': float(np.max(np.abs(audio))),
                'mean_amplitude': float(np.mean(np.abs(audio))),
                'rms_amplitude': float(np.sqrt(np.mean(audio**2)))
            },
            'top_frequencies': top_frequencies,
            'potential_horn_segments': potential_segments,
            'analysis_parameters': {
                'energy_threshold': float(energy_threshold),
                'centroid_mean': float(centroid_mean),
                'centroid_std': float(centroid_std),
                'total_horns_event': len(potential_segments)
            },
            'recommendations': {
                'suggested_amplitude_threshold': float(energy_threshold * 0.7),
                'suggested_freq_range': [500, 1500],
                'suggested_min_duration': 1
            }
        }
        
        # Save analysis
        with open(os.path.join(output_dir, 'audio_analysis_For_night_video.json'), 'w') as f:
            json.dump(analysis_report, f, indent=2)
        
        self.logger.info(f"Analysis complete! Found {len(potential_segments)} potential horn segments")
        self.logger.info(f"Analysis saved to {output_dir}/")
        
        return analysis_report
    
    def detect_horn_events_improved(self, audio, sr):
        """Improved horn detection using multiple signal processing techniques"""
        
        # 1. Normalize audio
        audio = librosa.util.normalize(audio)
        
        # 2. Multiple frequency band analysis
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = int(0.02 * sr)   # 20ms hop
        
        # Calculate multiple features
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # 3. Multi-criteria detection
        # Energy criterion
        energy_threshold = np.percentile(rms, 90)  # Top 10% energy
        energy_mask = rms > energy_threshold
        
        # Spectral characteristics (horns often have specific spectral patterns)
        centroid_mean = np.mean(spectral_centroids)
        centroid_std = np.std(spectral_centroids)
        
        # Look for spectral centroids in typical horn ranges (adjusted based on 8kHz sampling)
        horn_centroid_mask = (spectral_centroids > 200) & (spectral_centroids < 2000)
        
        # Combine criteria with weights
        combined_score = (energy_mask.astype(float) * 0.6 + 
                         horn_centroid_mask.astype(float) * 0.4)
        
        # Smooth the combined score
        combined_score = signal.savgol_filter(combined_score, 
                                            window_length=min(21, len(combined_score)//2*2+1), 
                                            polyorder=2)
        
        # Find peaks in combined score
        peaks, properties = signal.find_peaks(combined_score, 
                                            height=0.4,  # Lower threshold
                                            distance=int(0.3 * sr / hop_length))  # Min 300ms apart
        
        # Group consecutive detections into events
        horn_events = []
        if len(peaks) > 0:
            current_start = times[peaks[0]]
            current_end = times[peaks[0]]
            current_scores = [combined_score[peaks[0]]]
            
            for i in range(1, len(peaks)):
                peak_time = times[peaks[i]]
                if peak_time - current_end <= 0.5:  # If within 500ms, extend current event
                    current_end = peak_time
                    current_scores.append(combined_score[peaks[i]])
                else:
                    # End current event if it's long enough
                    if current_end - current_start >= self.config['min_horn_duration']:
                        horn_events.append({
                            'start_time': current_start,
                            'end_time': current_end,
                            'duration': current_end - current_start,
                            'peak_amplitude': np.max(rms[peaks[max(0, i-len(current_scores)):i]]),
                            'avg_score': np.mean(current_scores),
                            'max_score': np.max(current_scores)
                        })
                    current_start = peak_time
                    current_end = peak_time
                    current_scores = [combined_score[peaks[i]]]
            
            # Don't forget the last event
            if current_end - current_start >= self.config['min_horn_duration']:
                horn_events.append({
                    'start_time': current_start,
                    'end_time': current_end,
                    'duration': current_end - current_start,
                    'peak_amplitude': np.max(rms[peaks[-len(current_scores):]]),
                    'avg_score': np.mean(current_scores),
                    'max_score': np.max(current_scores)
                })
        
        return horn_events
    
    def extract_timestamp_from_frame(self, frame):
        """Extract timestamp from video frame - disabled for now"""
        return None  # OCR disabled as requested
    
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime object"""
        try:
            # Expected format: "12-31-2022 Sat 10:24:25"
            # Remove day name if present
            parts = timestamp_str.split()
            if len(parts) >= 3:
                date_part = parts[0]
                time_part = parts[-1]
                
                # Parse date and time
                date_obj = datetime.strptime(f"{date_part} {time_part}", "%m-%d-%Y %H:%M:%S")
                return date_obj
        except Exception as e:
            self.logger.error(f"Error parsing timestamp '{timestamp_str}': {e}")
        
        return None
    
    def process_video(self, video_path, output_path=None, analyze_only=False):
        """Process video file and detect horn events"""
        self.logger.info(f"Processing video: {video_path}")
        
        # Extract audio
        audio, sr = self.extract_audio_from_video(video_path)
        if audio is None:
            return None
        
        # If analysis mode, run detailed analysis first
        if analyze_only or self.config.get('analysis_mode', False):
            analysis_report = self.analyze_audio_characteristics(audio, sr)
            if analyze_only:
                return analysis_report
        
        # Detect horn events in audio using improved method
        horn_events = self.detect_horn_events_improved(audio, sr)
        self.logger.info(f"Detected {len(horn_events)} horn events")
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()
        
        # Process horn events and create results
        results = []
        for i, event in enumerate(horn_events):
            video_timestamp = event['start_time']
            
            result = {
                'event_id': i + 1,
                'video_timestamp_seconds': round(video_timestamp, 2),
                'video_timestamp_formatted': str(timedelta(seconds=int(video_timestamp))),
                'duration_seconds': round(event['duration'], 2),
                'peak_amplitude': round(float(event['peak_amplitude']), 4),
                'detection_score': round(float(event.get('max_score', 0)), 3),
                'confidence': self._calculate_confidence(event)
            }
            results.append(result)
        
        # Compile final output
        output = {
            'video_file': os.path.basename(video_path),
            'video_duration_seconds': round(video_duration, 2),
            'video_duration_formatted': str(timedelta(seconds=int(video_duration))),
            'total_horn_events': len(results),
            'processing_config': self.config,
            'horn_events': results
        }
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            self.logger.info(f"Results saved to: {output_path}")
        
        return output
    
    def _calculate_confidence(self, event):
        """Calculate confidence level for detected event"""
        score = event.get('max_score', 0)
        duration = event.get('duration', 0)
        amplitude = event.get('peak_amplitude', 0)
        
        # Simple confidence calculation
        if score > 0.7 and duration > 1.0 and amplitude > 0.01:
            return 'high'
        elif score > 0.5 and duration > 0.5:
            return 'medium'
        else:
            return 'low'

def main():
    parser = argparse.ArgumentParser(description='Railway Horn Detection from CCTV Footage')
    parser.add_argument('video_path', help='Path to MP4 video file')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--analyze', action='store_true', 
                       help='Run audio analysis only (helps tune parameters)')
    parser.add_argument('--threshold', type=float, default=0.001, 
                       help='Amplitude threshold for horn detection')
    parser.add_argument('--min-duration', type=float, default=0.3,
                       help='Minimum horn duration in seconds')
    
    args = parser.parse_args()
    
    # Install matplotlib if not available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found. Installing...")
        os.system("pip install matplotlib")
        import matplotlib.pyplot as plt
    
    # Custom configuration
    config = {
        'horn_freq_range': (500, 2000),
        'horn_peak_freq': (500, 1000),
        'min_horn_duration': args.min_duration,
        'amplitude_threshold': args.threshold,
        'analysis_mode': True,
    }
    
    # Initialize detector
    detector = RailwayHornDetector(config)
    
    if args.analyze:
        # Run analysis only
        print("Running audio analysis...")
        detector.process_video(args.video_path, analyze_only=True)
        print("Analysis complete! Check the 'analysis' folder for detailed results.")
        print("Use the recommendations in audio_analysis.json to tune detection parameters.")
    else:
        # Process video
        output_path = args.output or f"{os.path.splitext(args.video_path)[0]}_horn_detection.json"
        results = detector.process_video(args.video_path, output_path)
        
        if results:
            print(f"Processing complete!")
          #  print(f"Found {results['total_horn_events']} horn events")
            print(f"Video duration: {results['video_duration_seconds']} seconds")
            print(f"Results saved to: {output_path}")
            
            # Show some detected events
        #     if results['horn_events']:
        #         print("\nDetected Events:")
        #         for event in results['horn_events'][:5]:  # Show first 5
        #             print(f"  Event {event['event_id']}: {event['video_timestamp_formatted']} "
        #                   f"(duration: {event['duration_seconds']}s, confidence: {event['confidence']})")
        #         if len(results['horn_events']) > 5:
        #             print(f"  ... and {len(results['horn_events']) - 5} more events")
        # else:
        #     print("Processing failed!")

if __name__ == "__main__":
    main()