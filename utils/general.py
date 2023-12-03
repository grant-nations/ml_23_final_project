import os

def generate_unique_filename(filename):
    if not os.path.exists(filename):
        return filename

    name, ext = os.path.splitext(filename)
    counter = 1

    while True:
        new_filename = f"{name}-{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1
