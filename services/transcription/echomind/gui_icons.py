"""Small scalable line icons for Tk's native buttons, without font dependencies."""

import math
import tkinter as tk


def navigation_icon(master, name, size, color):
    paths = {
        "panel": [(3, 4, 21, 4, 21, 20, 3, 20, 3, 4), (9, 4, 9, 20)],
        "basic": [(4, 6, 20, 6), (4, 12, 20, 12), (4, 18, 20, 18),
                  (8, 3, 8, 9), (16, 9, 16, 15), (10, 15, 10, 21)],
        "speech": [(9, 4, 9, 13, 10, 15, 14, 15, 15, 13, 15, 4, 14, 2, 10, 2, 9, 4),
                   (5, 11, 5, 14, 7, 18, 12, 19, 17, 18, 19, 14, 19, 11), (12, 19, 12, 22), (8, 22, 16, 22)],
        "model": [(6, 6, 18, 6, 18, 18, 6, 18, 6, 6), (10, 10, 14, 10, 14, 14, 10, 14, 10, 10)]
                 + [(p, 3, p, 6, p, 3) for p in (9, 15)]
                 + [(p, 18, p, 21) for p in (9, 15)]
                 + [(3, p, 6, p) for p in (9, 15)] + [(18, p, 21, p) for p in (9, 15)],
        "jev": [(3, 4, 21, 4, 21, 17, 12, 17, 7, 21, 7, 17, 3, 17, 3, 4), (7, 10, 11, 13, 17, 7)],
        "settings": [(4, 6, 20, 6), (4, 12, 20, 12), (4, 18, 20, 18),
                     (8, 3, 8, 9), (16, 9, 16, 15), (10, 15, 10, 21)],
        "shortcut": [(2, 5, 22, 5, 22, 19, 2, 19, 2, 5), (6, 9, 7, 9), (11, 9, 12, 9),
                     (16, 9, 18, 9), (6, 14, 18, 14)],
        "follow": [(3, 3, 16, 3, 16, 7), (3, 3, 3, 16, 7, 16), (9, 9, 21, 9, 21, 21, 9, 21, 9, 9)],
        "update": [(5, 8, 12, 2, 19, 8), (12, 2, 12, 16), (3, 16, 3, 21, 21, 21, 21, 16)],
        "component": [(3, 7, 12, 2, 21, 7, 21, 17, 12, 22, 3, 17, 3, 7, 12, 12, 21, 7), (12, 12, 12, 22)],
    }
    image = tk.PhotoImage(master=master, width=size, height=size)
    scale = size / 24
    stroke = max(1, round(1.5 * scale))
    for path in paths[name]:
        points = list(zip(path[::2], path[1::2]))
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            steps = max(1, math.ceil(max(abs(x2 - x1), abs(y2 - y1)) * scale))
            for step in range(steps + 1):
                x = round((x1 + (x2 - x1) * step / steps) * scale)
                y = round((y1 + (y2 - y1) * step / steps) * scale)
                image.put(color, to=(x, y, min(size, x + stroke), min(size, y + stroke)))
    return image
