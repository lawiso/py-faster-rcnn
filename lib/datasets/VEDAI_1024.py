# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import datasets
from datasets.imdb import imdb
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from VEDAI_1024_eval import VEDAI_1024_eval
import collections
from fast_rcnn.config import cfg


class VEDAI_1024(imdb):

    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'VEDAI_1024' + '_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        # images data path (only positive)
        self._data_path = os.path.join(
            self._devkit_path, self._image_set, 'pos')
        self._classes = ('__background__',  # always index 0
                         'vehicle')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._get_image_list(self._data_path)
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'dedup_boxes': 0.25}

        assert os.path.exists(self._devkit_path), \
            'VEDAI_1024 path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        im_name = index + self._image_ext
        return os.path.join(self._data_path, im_name)

    def _get_default_path(self):
        """
        Return the default path where VEDAI_1024Person is expected to be installed.
        You need to add a link in the data folder to VEDAI_1024Person datasets folder
        """
        return os.path.join(cfg.DATA_DIR, 'VEDAI_1024')

    def _get_image_list(self, data_path):
        """
        Return a List that contains all images
        """
        ims = []
        for root, dirs, files in os.walk(data_path):
            ims = [os.path.splitext(file)[0] for file in files]
        return ims

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_VEDAI_1024_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + cfg.TEST.BBOX_METHODE + '_roidb.pkl')
        if cfg.USE_SS_CACHE:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    roidb = cPickle.load(fid)
                print '{} ss roidb loaded from {}'.format(self.name, cache_file)
                return roidb
        if cfg.TEST.BBOX_METHODE == 'rpn_s1':
            self.config['rpn_file'] = cfg.TEST.RPN_FILE
            return self.rpn_roidb()
        if self._image_set != 'Test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'Test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        #from IPython import embed; embed()
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        base_path = os.path.join(cfg.DATA_DIR, 'VEDAI_1024')
        if self._image_set == 'Test':
            filename = os.path.join(base_path, 'Test','proposals', self.get_test_proposals())
        else:
            filename = os.path.join(base_path, 'Train','proposals', 'train.mat')

        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()
        names = sio.loadmat(filename)['images'].ravel()


        d = {}
        for i, name in enumerate(names):
            #print name[0][0]
            #n = os.path.splitext(name[0])[0]
            n = name[0]
            #print n
            idx = self._image_index.index(n)
            d[idx] = i
        od = collections.OrderedDict(sorted(d.items()))
        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[od[i]][:, (1, 0, 3, 2)]
            num_boxes = boxes.shape[0]
            num_boxes_should = cfg.TEST.PROPOSAL_NUMBER
            if num_boxes > num_boxes_should: 
                keep = np.random.randint(num_boxes, size = num_boxes_should)
                boxes = boxes[keep, :]            
            #from IPython import embed
            #embed()
            box_list.append(boxes)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_VEDAI_1024_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, '..', 'annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        #if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
        #    non_diff_objs = [
        #        obj for obj in objs if int(obj.find('difficult').text) == 0]
        #    if len(non_diff_objs) != len(objs):
        #        print 'Removed {} difficult objects'.format(
        #            len(objs) - len(non_diff_objs))
        #    objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area        
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_VEDAI_1024_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + \
            self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _write_VEDAI_1024_results_file(self, all_boxes):
        #print('Debug -- Shape of all_boxes {}'.format(all_boxes))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VEDAI_1024 results file'.format(cls)
            filename = self._get_VEDAI_1024_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        #print('Debug -- dets is null')
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        toWrite = '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                            index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1)
                        #print('Debug -- toWrite'.format(toWrite))
                        #raw_input('to continue')
                        f.write(toWrite)

    def _do_python_eval(self, output_dir='output'):
        print '--------------------------------------------------------------'
        print 'Computing results for VEDAI_1024Person dataset.'
        print '--------------------------------------------------------------'
        annopath = os.path.join(
            self._devkit_path,
            self._image_set,
            'annotations',
            '{:s}.xml')
        imageset = self._image_index
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        #use_07_metric = True if int(self._year) < 2010 else False
        use_07_metric = False
        # print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_VEDAI_1024_results_file_template().format(cls)
            rec, prec, ap = VEDAI_1024_eval(
                filename, annopath, imageset, cls, cachedir, ovthresh=cfg.TEST.AP_THRESHOLD,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(datasets.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_VEDAI_1024_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_VEDAI_1024_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.VEDAI_1024 import VEDAI_1024
    d = VEDAI_1024('Train')
    res = d.roidb
    from IPython import embed
    embed()
