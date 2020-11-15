def get_resize_x_y(x, y, resize_factor):
    return int(x * resize_factor / 2) * 2, int(y * resize_factor / 2) * 2
