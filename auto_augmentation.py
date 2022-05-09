import Augmentor
# Install terlebih dahulu library augmentasi dataset gambar
# pip install augmentor / !pip install augmentor / %pip install augmentor


# Direktori/folder gambar yang akan di augmentasi
# Hasil augmentasi juga akan masuk ke dalam folder ini (Otomatis membuat sub-folder 'output')
p = Augmentor.Pipeline(r"F:\PyCharm-OD\Plat-Nomor\Registered\RegisteredB")

# Proses Augmentasi gambar
p.zoom(probability = 0.3, min_factor = 0.5, max_factor = 1.5) # Memperbesar gambar
p.flip_top_bottom(probability = 0.4) # Flip/mirror gambar atas-bawah
p.flip_left_right(probability = 0.4) # Flip/mirror gambar kanan-kiri
#p.random_brightness(probability=0.3, min_factor=0.3, max_factor=2) # Mengatur brightness/pencahayaan secara random
p.random_brightness(probability = 0.3, min_factor = 0.2, max_factor = 2) # Mengatur brightness/pencahayaan secara random
p.random_distortion(probability = 1, grid_width = 4, grid_height = 4, magnitude = 7) # Mengatur distorsi gambar secara random

# Banyak gambar yang akan di-generate (di augmentasi)
p.sample(1000)