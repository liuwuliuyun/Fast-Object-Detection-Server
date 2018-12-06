
from detect import image_prepare, detect

upload_path = 'static/images/dog.jpg'

input_imgs = image_prepare(upload_path)
print(input_imgs.shape)
infer_time = detect(input_imgs, upload_path)