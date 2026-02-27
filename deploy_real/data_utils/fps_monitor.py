"""
FPS monitoring utility for measuring and tracking loop execution frequency.
"""
import time
import numpy as np


class FPSMonitor:
    """
    Monitor and track FPS (frames per second) of a running loop.
    
    Features:
    - Quick FPS reporting every N steps (default 100)
    - Detailed statistics every M steps (default 1000) when enabled
    - Tracks average, min, max, and std deviation of FPS
    """
    
    def __init__(self, 
                 enable_detailed_stats=False,
                 quick_print_interval=100,
                 detailed_print_interval=1000,
                 expected_fps=None,
                 name="Loop"):
        """
        Initialize FPS monitor.
        
        Args:
            enable_detailed_stats: Whether to print detailed statistics
            quick_print_interval: Print quick stats every N steps
            detailed_print_interval: Print detailed stats every M steps (if enabled)
            expected_fps: Expected FPS for comparison (optional)
            name: Name of the loop being monitored (for logging)
        """
        self.enable_detailed_stats = enable_detailed_stats
        self.quick_print_interval = quick_print_interval
        self.detailed_print_interval = detailed_print_interval
        self.expected_fps = expected_fps
        self.name = name
        
        # Tracking variables
        self.last_time = None
        self.execution_times = []
        self.step_count = 0
        
        # Detailed stats tracking
        self.fps_measurements = []
        self.detailed_iteration_count = 0
        
    def tick(self):
        """
        Call this at each loop iteration to record timing.
        Automatically prints statistics at configured intervals.
        """
        current_time = time.time()
        
        if self.last_time is not None:
            interval = current_time - self.last_time
            current_fps = 1.0 / interval if interval > 0 else 0.0
            
            # Track for quick printing
            self.execution_times.append(interval)
            self.step_count += 1
            
            # Print quick stats
            if self.step_count % self.quick_print_interval == 0:
                self._print_quick_stats()
            
            # Track and print detailed stats if enabled
            if self.enable_detailed_stats:
                self.fps_measurements.append(current_fps)
                self.detailed_iteration_count += 1
                
                if self.detailed_iteration_count == self.detailed_print_interval:
                    self._print_detailed_stats()
                    # Reset for next measurement period
                    self.fps_measurements = []
                    self.detailed_iteration_count = 0
        
        self.last_time = current_time
    
    def _print_quick_stats(self):
        """Print quick FPS statistics based on recent measurements."""
        recent_intervals = self.execution_times[-self.quick_print_interval:]
        avg_interval = np.mean(recent_intervals)
        avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
        
        print(f"{self.name} Execution FPS (last {self.quick_print_interval} steps): "
              f"{avg_fps:.2f} Hz (avg interval: {avg_interval*1000:.2f}ms)")
    
    def _print_detailed_stats(self):
        """Print detailed FPS statistics."""
        avg_fps = np.mean(self.fps_measurements)
        max_fps = np.max(self.fps_measurements)
        min_fps = np.min(self.fps_measurements)
        std_fps = np.std(self.fps_measurements)
        
        start_step = self.step_count - self.detailed_print_interval + 1
        end_step = self.step_count
        
        print(f"\n{'='*85}")
        print(f"{self.name} Execution FPS Results (steps {start_step}-{end_step})")
        print(f"{'='*85}")
        print(f"Average FPS: {avg_fps:.2f} Hz")
        print(f"Max FPS:     {max_fps:.2f} Hz")
        print(f"Min FPS:     {min_fps:.2f} Hz")
        print(f"Std FPS:     {std_fps:.2f} Hz")
        if self.expected_fps is not None:
            print(f"Expected FPS: {self.expected_fps:.2f} Hz")
            print(f"Deviation:    {avg_fps - self.expected_fps:+.2f} Hz ({(avg_fps/self.expected_fps - 1)*100:+.1f}%)")
        print(f"{'='*85}\n")
    
    def get_current_fps(self):
        """
        Get current instantaneous FPS.
        
        Returns:
            Current FPS or None if not enough data
        """
        if len(self.execution_times) > 0:
            recent_interval = self.execution_times[-1]
            return 1.0 / recent_interval if recent_interval > 0 else 0.0
        return None
    
    def get_average_fps(self, window_size=None):
        """
        Get average FPS over recent measurements.
        
        Args:
            window_size: Number of recent measurements to average (None = all)
            
        Returns:
            Average FPS or None if not enough data
        """
        if len(self.execution_times) == 0:
            return None
            
        if window_size is None:
            recent_intervals = self.execution_times
        else:
            recent_intervals = self.execution_times[-window_size:]
        
        avg_interval = np.mean(recent_intervals)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    def reset(self):
        """Reset all measurements."""
        self.last_time = None
        self.execution_times = []
        self.step_count = 0
        self.fps_measurements = []
        self.detailed_iteration_count = 0

