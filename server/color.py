import webcolors

def closest_color(requested_rgb):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        # Euclidean distance
        dist = (r_c - requested_rgb[0]) ** 2 + (g_c - requested_rgb[1]) ** 2 + (b_c - requested_rgb[2]) ** 2
        min_colors[dist] = name
    return min_colors[min(min_colors.keys())]
