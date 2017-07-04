
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util


def read_label_file(label_file_path):
    object = []
    with open(label_file_path) as label_file:
        raw_lines = [line.strip() for line in label_file.readlines()]
        for raw_line in raw_lines:
            class_num, c_x, c_y, w, h = [float(e) for e in raw_line.split(" ")]
            x1 = (c_x - w / 2)
            y1 = (c_y - h / 2)
            x2 = (c_x + w / 2)
            y2 = (c_y + h / 2)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, 1)
            y2 = min(y2, 1)
            class_num = int(class_num)
            object.append([class_num, x1, y1, x2, y2])
    return object


def main():
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    # image_idx = 0
    image_list_path = r"F:\Database\bus_passenger_count\data_set\train\train_bk.txt"
    # image_list_path = r"F:\Database\bus_passenger_count\data_set\validate\val.txt"
    writer = tf.python_io.TFRecordWriter("F:\\tensorflow\\tfrecord\\head_train.record")
    # writer = tf.python_io.TFRecordWriter("F:\\tensorflow\\tfrecord\\head_val.record")
    with open(image_list_path, "r") as file:
        image_list = [line.strip().split() for line in file.readlines()]
        for img_path in image_list:
            # print(img_path[0])
            with tf.gfile.GFile(img_path[0], 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            # image = PIL.Image.open(img_path[0])
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            key = hashlib.sha256(encoded_jpg).hexdigest()
            width = image.width
            height = image.height
            # print(width, height)
            label_path = img_path[0].replace("images", "labels").replace("jpg", "txt")
            object = read_label_file(label_path)
            # print(len(object))
            for obj_num in range(0, len(object)):
                xmin.append(object[obj_num][1])
                ymin.append(object[obj_num][2])
                xmax.append(object[obj_num][3])
                ymax.append(object[obj_num][4])
                classes_text.append('head'.encode('utf8'))
                classes.append(object[obj_num][0] + 1)  # 类别从1开始
                difficult_obj.append(0)
                truncated.append(1)
                poses.append('Unspecified'.encode('utf8'))
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(
                    img_path[0].strip().split('/')[-1].encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(
                     img_path[0].strip().split('/')[-1].encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
                'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
                'image/object/truncated': dataset_util.int64_list_feature(truncated),
                'image/object/view': dataset_util.bytes_list_feature(poses),
            }))
            # image_idx +=1
            # if image_idx == 1:
            #     print(example)
            writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
  main()
