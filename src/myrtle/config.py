LOG_DIRECTORY = "myrtle_logs"

# MQ_HOST = "192.168.1.10"  # loki
MQ_HOST = "192.168.1.20"  # aidas
MQ_PORT = 38388  # An arbitrary port

WINDOWS_LEFT_PIXEL = 0
WINDOWS_TOP_PIXEL = 0
WINDOWS_WIDTH_PIXELS = 1800
WINDOWS_HEIGHT_PIXELS = 1000
WINDOW_TITLE_HEIGHT = 100
BENCH_HEIGHT_FRACTION = 0.35

"""
window_pixels (tuple(int))
The location of the windows for myrtle's dashboard constellation.
    * x-pixel of the left edge of the window
    * y-pixel of the top edge of the window
        (counting down from the top of the screen)
    * width of the window in pixels
    * height of the window in pixels
"""
# window_pixels = (x, y, width, height)
window_pixels = (
    WINDOWS_LEFT_PIXEL,
    WINDOWS_TOP_PIXEL,
    WINDOWS_WIDTH_PIXELS,
    WINDOWS_HEIGHT_PIXELS,
)
x, y, width, height = window_pixels
bench_height = int(height * BENCH_HEIGHT_FRACTION)  # - WINDOW_TITLE_HEIGHT
half_width = int(width / 2)
bench_window = (x, y, half_width, bench_height)
world_window = (
    x,
    y + bench_height + WINDOW_TITLE_HEIGHT,
    half_width,
    height - bench_height - WINDOW_TITLE_HEIGHT,
)
agent_window = (x + half_width, y, half_width, height - WINDOW_TITLE_HEIGHT)
