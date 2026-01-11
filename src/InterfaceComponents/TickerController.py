"""Ticker and animation control for the compiler UI.

Manages auto-progress speed, pause/resume, and timing control.
Isolates animation concerns from the main app class.
"""

from typing import Callable


class TickerController:
    """Manages auto-progress ticker (speed, pause/resume).
    
    Responsibilities:
    - Ticker interval management
    - Pause/resume control
    - Speed adjustment (increase/decrease)
    - Running state tracking
    """

    # Configuration constants
    DEFAULT_INTERVAL = 1.0
    MIN_INTERVAL = 0.1
    MAX_INTERVAL = 5.0
    SPEED_DELTA = 0.1

    def __init__(self, app):
        """Initialize the ticker controller.
        
        Args:
            app: Reference to the main app for ticker management
        """
        self.app = app
        self.interval = self.DEFAULT_INTERVAL
        self.running = False
        self._ticker = None
        self._tick_callback = None

    def start(self, tick_callback: Callable) -> None:
        """Start the ticker with a callback.
        
        Args:
            tick_callback: Callback function to call on each tick
        """
        self._tick_callback = tick_callback
        if self._ticker:
            self._ticker.stop()
        self._ticker = self.app.set_interval(
            self.interval,
            tick_callback,
            pause=not self.running
        )

    def pause(self) -> None:
        """Pause the ticker."""
        self.running = False
        if self._ticker:
            self._ticker.pause()

    def resume(self) -> None:
        """Resume the ticker."""
        self.running = True
        if self._ticker:
            self._ticker.resume()

    def toggle(self) -> bool:
        """Toggle running state.
        
        Returns:
            bool: New running state
        """
        if self.running:
            self.pause()
        else:
            self.resume()
        return self.running

    def increase_speed(self) -> float:
        """Increase speed (decrease interval).
        
        Returns:
            float: New interval value
        """
        self.interval = max(self.MIN_INTERVAL, self.interval - self.SPEED_DELTA)
        self._restart_ticker()
        return self.interval

    def decrease_speed(self) -> float:
        """Decrease speed (increase interval).
        
        Returns:
            float: New interval value
        """
        self.interval = min(self.MAX_INTERVAL, self.interval + self.SPEED_DELTA)
        self._restart_ticker()
        return self.interval

    def set_interval(self, interval: float) -> None:
        """Set the ticker interval directly.
        
        Args:
            interval: New interval in seconds
        """
        interval = max(self.MIN_INTERVAL, min(self.MAX_INTERVAL, interval))
        self.interval = interval
        self._restart_ticker()

    def get_interval(self) -> float:
        """Get the current ticker interval.
        
        Returns:
            float: Current interval in seconds
        """
        return self.interval

    def is_running(self) -> bool:
        """Check if ticker is running.
        
        Returns:
            bool: True if ticker is running
        """
        return self.running

    def _restart_ticker(self) -> None:
        """Restart ticker with current interval and preserve running state."""
        if self._ticker and self._tick_callback:
            was_running = self.running
            self._ticker.stop()
            # Recreate ticker with new interval
            self._ticker = self.app.set_interval(
                self.interval,
                self._tick_callback,
                pause=not was_running
            )
            self.running = was_running
