import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import collections #import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from res_object_detection.utils import ops as utils_ops
from res_object_detection.utils import label_map_util

import abc
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
# import PIL.Image as Image
# import PIL.ImageColor as ImageColor
# import PIL.ImageDraw as ImageDraw
# import PIL.ImageFont as ImageFont
import six
from six.moves import range
from six.moves import zip
#import tensorflow.compat.v1 as tf

from object_detection.core import keypoint_ops
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils

#from research.object_detection.utils import visualization_utils as vis_util

im_height = 480
im_width = 640

def unnormalize(box, legend=False):
    if legend:
        safe_zone = 0
    else:
        safe_zone = 8
    ymin, xmin, ymax, xmax = box
    return ((xmin * im_width)-safe_zone, (ymin * im_height)-safe_zone, (xmax * im_width)+safe_zone, (ymax * im_height)+safe_zone)

def load_model():
    model = tf.saved_model.load('fine_tuned_model/saved_model')
    model = model.signatures['serving_default']
    return model

PATH_TO_LABELS = 'label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg"))) #note png / jpg difference
TEST_IMAGE_PATHS

#model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
detection_model = load_model()

detection_model.output_dtypes
detection_model.output_shapes


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def record_boxes(image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.2,
    agnostic_mode=False,
    # line_thickness=4,
    # groundtruth_box_visualization_color='black',
    # skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    #skip_track_ids=False
    ):
  """
  group boxes together that correspond ot the same location
  """
  box_dic = {}
  box_to_class_score_map = {}
  
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_dic):
      break
    if scores is None or scores[i] > min_score_thresh:
        box = tuple(boxes[i].tolist())
        display_class = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in six.viewkeys(category_index):
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_class = str(class_name)
        if not skip_scores:
          display_score = round(100*scores[i])

        if display_class not in box_dic:
          box_dic[display_class] = set()
        if display_class == 'legend':
            box_dic[display_class].add((unnormalize(box, legend=True), display_score))
        else:
            box_dic[display_class].add((unnormalize(box), display_score))

  box_dic['image_height'] = image.shape[0]
  box_dic['image_width'] = image.shape[1]
  # print(box_dic['image_height'])
  # print(box_dic['image_width'])
  #print(box_dic)
  # for k in box_dic:
  #     if k != 'image_width' and k != 'image_height':
  #       for elem in box_dic[k]:
  #           (box, score) = elem
            # print(k)
            # print(box)
            # print(score)
  return box_dic
      

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  return record_boxes(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True)



def assign_labels(d):
    labels = {}
    x_axis_label = None
    y_axis_label = None#'no y axis'
    title_label = None#'no title'
    legend = None#'no legend'
    img_mid_x = d['image_width']/2
    #img_mid_y = d['image_height']/2
    if 'text' in d:
        max_x_displacement = 0
        min_y = d['image_height'] # because y is inverted
        for elem in d['text']:
            box,score = elem
            cand = max(abs(box[0]-img_mid_x), abs(box[2]-img_mid_x))
            if cand > max_x_displacement:
                max_x_displacement = cand
                y_axis_label = box
            if box[3] < min_y:
                min_y = box[3]
                title_label = box
        for i,elem in enumerate(d['text']):
            box, score = elem
            if box != y_axis_label and box != title_label:
                x_axis_label = box
                #labels[box] = 'x axis'
                #print("x success")
    labels[x_axis_label] = 'x axis'
    labels[y_axis_label] = 'y axis'
    labels[title_label] = 'title'
    # print(title_label)
    # print(y_axis_label)
    if 'legend' in d:
        for elem in d['legend']:
            box,score = elem
            legend = box
    labels[legend] = 'legend' # takes the last legend if multiple, but shouldn't be an issue since there should only ever be one
    return labels

    


        

#   print(output_dict.keys())
#   print(output_dict['num_detections'])
  #output_dict['detection_boxes']

def predict():
    predictions = []
    for image_path in TEST_IMAGE_PATHS:
        predictions.append(show_inference(detection_model, image_path))
    return predictions

single_img_path = "images/test.jpg" #PATH_TO_TEST_IMAGES_DIR.glob("test.jpg")
# print(single_img_path)
# print(assign_labels(show_inference(detection_model, single_img_path)))

#show_inference(detection_model, single_img_path, 1)


#   # Visualization of the results of a detection.
  
  
#   vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       output_dict['detection_boxes'],
#       output_dict['detection_classes'],
#       output_dict['detection_scores'],
#       category_index,
#       instance_masks=output_dict.get('detection_masks_reframed', None),
#       use_normalized_coordinates=True,
#       line_thickness=8)
# #   plt.figure()
# #   plt.imshow(Image.fromarray(image_np))
# #   plt.savefig('testod.png')
#   #display(Image.fromarray(image_np))
#   Image.fromarray(image_np).save('predictions/testod'+ str(i) +'.png')

'''
Notes form vis/vis function
'''
#box_to_display_str_map = collections.defaultdict(list)
  #box_to_color_map = collections.defaultdict(str)
  #box_to_instance_masks_map = {}
  #box_to_instance_boundaries_map = {}
  #box_to_keypoints_map = collections.defaultdict(list)
  #box_to_keypoint_scores_map = collections.defaultdict(list)
  #box_to_track_ids_map = {}
 #   if instance_masks is not None:
    #     box_to_instance_masks_map[box] = instance_masks[i]
    #   if instance_boundaries is not None:
    #     box_to_instance_boundaries_map[box] = instance_boundaries[i]
    #   if keypoints is not None:
    #     box_to_keypoints_map[box].extend(keypoints[i])
    #   if keypoint_scores is not None:
    #     box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
    #   if track_ids is not None:
    #     box_to_track_ids_map[box] = track_ids[i]
    #   if scores is None:
    #     box_to_color_map[box] = groundtruth_box_visualization_color
    #   else:
        

        #   if not display_str:
        #     display_score = '{}%'.format(round(100*scores[i]))
        #     #display_str = '{}%'.format(round(100*scores[i]))
        #   else:
        #     #display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
        #     displ
        # if not skip_track_ids and track_ids is not None:
        #   if not display_str:
        #     display_str = 'ID {}'.format(track_ids[i])
        #   else:
        #     display_str = '{}: ID {}'.format(display_str, track_ids[i])
        #box_to_class_and_score_map[box].append((display_class, display_score))
        # if agnostic_mode:
        #   box_to_color_map[box] = 'DarkOrange'
        # elif track_ids is not None:
        #   prime_multipler = _get_multiplier_for_color_randomness()
        #   box_to_color_map[box] = STANDARD_COLORS[
        #       (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        # else:
        #   box_to_color_map[box] = STANDARD_COLORS[
        #       classes[i] % len(STANDARD_COLORS)]

#   # record coords
#   for box, st in box_to_display_str_map.items():
#       ymin, xmin, ymax, xmax = box
#       print(st[0])
#       #print(st[1])
#       if st[0] not in box_dic:
#           box_dic[st[0]] = set()
#       box_dic[st[0]].add(box)
  