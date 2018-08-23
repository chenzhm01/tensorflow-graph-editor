#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 08:54:19 2018

@author: commaai02
"""

import tensorflow as tf
import cv2
import numpy as np

box_pb_path='./faster_inception.pb-30000'
ge = tf.contrib.graph_editor


def load_0(g, pb_path, sess):
    with sess.as_default():
      with g.as_default():
        gx = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
          serialized_graph = fid.read()
          gx.ParseFromString(serialized_graph)
          tf.import_graph_def(gx, name='')
        _inputs = g.get_tensor_by_name('image_tensor:0')
        _scores = tf.squeeze(g.get_tensor_by_name('detection_scores:0'), [0])
        _boxes = tf.squeeze(g.get_tensor_by_name('detection_boxes:0'), [0])
        _classes = tf.squeeze(g.get_tensor_by_name('detection_classes:0'), [0])
        
        return _inputs,_boxes,_classes,_scores

def load_1(g, pb_path, sess):
    with sess.as_default():
      with g.as_default():
        gx = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
          serialized_graph = fid.read()
          gx.ParseFromString(serialized_graph)
          tf.import_graph_def(gx, name='')
        _inputs = g.get_tensor_by_name('image_tensor:0')
        _scores = tf.squeeze(g.get_tensor_by_name('detection_scores:0'), [0])
        _boxes = tf.squeeze(g.get_tensor_by_name('detection_boxes:0'), [0])
        _classes = tf.squeeze(g.get_tensor_by_name('detection_classes:0'), [0])
        
        inputs = [_inputs.op]
        outputs = [_scores.op, _boxes.op, _classes.op]
        
        wbo = ge.get_within_boundary_ops(tf.get_default_graph(), inputs, outputs)
        sgv = ge.make_view(wbo)
        
        another_inputs = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name='xx')
        new_sgv, info = ge.copy_with_input_replacements(sgv, 
                                                        {_inputs: another_inputs},
                                                        dst_scope='replace')

        new_inputs = info.transformed(_inputs)
        new_scores = info.transformed(_scores)
        new_boxes = info.transformed(_boxes)
        new_classes = info.transformed(_classes)
        
        return new_inputs, new_boxes, new_classes, new_scores

 
def reset_inputs():
    image = cv2.imread('./image2.jpg')
    image = np.expand_dims(image, axis=0)
    image_tensor = tf.constant(image)
    return image_tensor
    
def load_2(g, pb_path, sess):
    with sess.as_default():
      with g.as_default():
        gx = tf.GraphDef()
        new_inputs = reset_inputs()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
          serialized_graph = fid.read()
          gx.ParseFromString(serialized_graph)
          tf.import_graph_def(gx, input_map={"image_tensor:0": new_inputs}, name='')
          #tf.import_graph_def(gx, name='')
        #_inputs = g.get_tensor_by_name('image_tensor:0')
        _scores = tf.squeeze(g.get_tensor_by_name('detection_scores:0'), [0])
        _boxes = tf.squeeze(g.get_tensor_by_name('detection_boxes:0'), [0])
        _classes = tf.squeeze(g.get_tensor_by_name('detection_classes:0'), [0])
        
        return _boxes,_classes,_scores
        

def main():  
    g=tf.Graph()
    sess = tf.Session(graph=g)
    #step0_inputs,step0_boxes,step0_classes,step0_scores = load_0(g, box_pb_path,sess)
    #step1_inputs,step1_boxes,step1_classes,step1_scores = load_1(g, box_pb_path,sess)
    step2_boxes,step2_classes,step2_scores = load_2(g, box_pb_path,sess)
    #print('step0_inputs: ',step0_inputs)
    #print('step1_inputs: ',step1_inputs)

    box, cls, score = sess.run([step2_boxes,step2_classes,step2_scores])
    return box, cls, score

if __name__ == '__main__':
    box, cls, score = main()
    image = cv2.imread('./image/image2.jpg')
    h,w,_ = image.shape
    xmin = int(box[0][1]*w)
    ymin = int(box[0][0]*h)
    xmax = int(box[0][3]*w)
    ymax = int(box[0][2]*h)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    
    while(1):
      cv2.imshow('test',image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
         break
    cv2.destroyAllWindows()
    
    
