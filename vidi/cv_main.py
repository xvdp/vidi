"""wrapper class for all cv manipulations"""
import os
import os.path as osp
import json
import time
import copy
import numpy as np
import cv2
import cvui
from .io_main import IO

MAIN_WIN = 'VIDI'

class CV():
    """wrapper to cv2 functions"""
    def __init__(self, fname=None, template=None):
        
        # io class needs cleanup - 
        # self.file, self.io.file, self.template, self.io.template are all the same        
        self.io = IO(fname)

        # cv capture object
        self.vid = None

        # read numpy array
        self.image = None
        # temp array for display
        self.processed_image = None

        # annotation template
        self.template = template
        # dictionary of annotation template copies, per frame
        self.features = {}
        # dictionary of clean annotations  per frame
        self.out_features = {}

        self.debug = False
        self.stats = {}

        self.first = 0
        self.last = 0

        self.current = 0

        self.wait_ms = 0
        self.store_ms = 0

        self.scale = 1
        self.handle_edge = 1
        self.edges = False
        self.direction = 1

        self.ispaused = False

        self.mouse_data = None
        self.getting_feature = False

        self.rectangle = cvui.Rect(0, 0, 0, 0)

    def __delete__(self, instance):
        if self.vid is not None:
            self._close()

    def _open(self, fname=None):

        self.io.file_resolve(fname)
        if self.io.file is None:
            return None
        try:
            print(self.io.file)
            self.vid = None
            self.vid = cv2.VideoCapture(self.io.file)
            self._stats()

            cv2.startWindowThread()
            # cvui.init(MAIN_WIN)
            cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(MAIN_WIN, self._width, self._height)
            print("playing with opencv <%s>"%(cv2.__version__))
            print("------------------------")
            print("keys: 'q' quit")
            print("      '+' increase playback (max 1000fps)")
            print("      '-' decrease playback (min 1fps)")
            print("      '0' set original framerate")
            print("      'd' show trackbar")
            print("      's' half display size")
            print("      'l' double display size")
            print("      'o' original image size")
            print("      TAB single frame")
            print("      SPACE pause")
            print("------------------------")
            return True
        except:
            print("cant open", self.io.file)
            return None

    def _open_img(self):
        try:
            self.image = cv2.imread(self.io.file, 1)
            cv2.startWindowThread()
            #self._prepare_image()
            return self.image
        except:
            print("cant open", self.io.file)
            return None

    def _close(self):
        cv2.waitKey(1)
        self.vid.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        self.vid = None

    def get_stats(self, fname=None):
        if self.vid is None:
            ret = self._open()
            if ret is None:
                print('valid video file missing ..', fname)
                return None

        self._stats()
        self._close()


    def _stats(self):
        if self.image is None:
            if self.debug:
                print('reading image frame')
            _, self.image = self.vid.read()

        self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.last == 0:
            self.last = self.frames - 1

        self.rate = self.vid.get(cv2.CAP_PROP_FPS)
        self.format = self.vid.get(cv2.CAP_PROP_FORMAT)
        self.fourcc = self.vid.get(cv2.CAP_PROP_FOURCC)
        self.shape = self.image.shape
        self.height = self.shape[0]
        self.width = self.shape[1]
        self._height = self.height
        self._width = self.width

        self.ms_frame_ = 1000./float(self.rate)
        self.ms_frame = int(self.ms_frame_)

        # integer frame rate
        self.wait_ms = self.ms_frame
        self.store_ms = self.ms_frame

        self.pad = "%%0%dd" %len(str(self.frames))

        self.stats['width'] = self.width
        self.stats['height'] = self.height
        self.stats['frames'] = int(self.frames)
        self.stats['rate'] = self.rate
        self.stats['ms_frame'] = self.ms_frame_
        self.stats['current'] = self.current
        self.stats['pad'] = self.pad
        self.stats['codec'] = self.fourcc
        self.stats['format'] = self.format
        self.stats['file'] = self.io.file
        print(json.dumps(self.stats, indent=2))

    def play(self, fname=None):
        """ff play video"""
        self._open(fname)

        ret = True
        while ret:

            #read video
            if not self.ispaused:
                ret, self.image = self.vid.read()

            #insert here other file operations tbd
            self.processed_image = self.image.copy()

            #markup ui
            if self.ispaused:
                self.ui_build()

            if not self.keyhandler(cv2.waitKey(self.wait_ms) & 0xFF):
                break
            cv2.imshow(MAIN_WIN, self.processed_image)
        self._close()

    def trackbar(self, trackname='track'):
        def onchange(x):
            if self.wait_ms == 0:
                pos = cv2.getTrackbarPos('track', MAIN_WIN)
                #self._get_image(pos)
                self._set_frame(pos)

                _, self.image = self.vid.read()
                cv2.imshow(MAIN_WIN, self.image)
            else:
                cv2.setTrackbarPos('track', MAIN_WIN, int(self.current))
        cv2.createTrackbar(trackname, MAIN_WIN, int(self.current), int(self.frames), onchange)


    def pause(self):
        """pauses loop, full pause, no read no redraw"""
        if self.wait_ms == 0:
            self.wait_ms = self.store_ms
        else:
            self.store_ms = self.wait_ms
            self.wait_ms = 0
        return self.wait_ms


    def ui_text(self, text):
        cvui.text(self.processed_image, 40, 10, text)

    def ui_feature(self, dic):

        self.ui_text(text='Click Button to create feature')
        for name in dic.keys():

            cvui.checkbox(self.processed_image, 20, self.ui_y, name, dic[name]['check'])

            if dic[name]['data']:
                if dic[name]['type'] == "line":
                    if len(dic[name]['data']) == 4:
                        cv2.line(self.processed_image, tuple(dic[name]['data'][:2]), tuple(dic[name]['data'][2:]), (0, 255, 255), 3)
                elif dic[name]['type'] == 'area':
                    if len(dic[name]['data']) == 4:
                        cvui.rect(self.processed_image, *dic[name]['data'], 0xff0000)
                elif dic[name]['type'] == 'areas':
                    if dic[name]['data']:
                        for d in dic[name]['data']:
                            if len(d) == 4:
                                cvui.rect(self.processed_image, *d, 0xff0000)
                elif dic[name]['type'] == "marker" and len(dic[name]['data'])==2:
                    cv2.circle(self.processed_image, dic[name]['data'], 3, (0, 255, 255), -1)
                else:
                    print('feature type', dic[name]['type'], 'not recognized')


            self.ui_y += 20

    def ui_checkmouse(self, dic):
        """ use checkbox to trigger feature
            how to fix features check again.
        """
        for name in dic.keys():
            if dic[name]['check'][0]:
                dlen = len(dic[name]['data'])
                ntype = dic[name]['type']

                if (dlen == 2 and ntype == 'marker') or (ntype in ['area', 'line'] and dlen == 4):
                    dic[name]['data'] = []

                if cvui.mouse(cvui.DOWN):
                    #plurarl areas
                    if dic[name]['type'] == 'areas':
                        dic[name]['data'].append([cvui.mouse().x, cvui.mouse().y])

                    #singluar area, marker, line
                    else:
                        dic[name]['data'] = [cvui.mouse().x, cvui.mouse().y]

                        # markers need only one exit after
                        if dic[name]['type'] == 'marker':
                            print(dic[name]['type'], dic[name]['data'])
                            dic[name]['check'][0] = False

                    #temporary draw
                    if dic[name]['type'] in ['area', 'areas']:
                        self.rectangle.x = cvui.mouse().x
                        self.rectangle.y = cvui.mouse().y

                if cvui.mouse(cvui.IS_DOWN):
                    #temporary draw
                    if dic[name]['type'] == 'line':
                        cv2.line(self.processed_image, tuple(dic[name]['data']), (cvui.mouse().x, cvui.mouse().y),(0,0,255), 3)

                    if dic[name]['type'] in ['area', 'areas']:
                        # Adjust rectangle dimensions according to mouse pointer
                        self.rectangle.width = cvui.mouse().x - self.rectangle.x
                        self.rectangle.height = cvui.mouse().y - self.rectangle.y
                        #print('area(s)', self.rectangle.x, self.rectangle.y, self.rectangle.width, self.rectangle.height)
                        cvui.printf(self.processed_image, self.rectangle.x + 5, self.rectangle.y + 5, 0.3, 0xff0000, '(%d,%d)', self.rectangle.x, self.rectangle.y)
                        cvui.printf(self.processed_image, cvui.mouse().x + 5, cvui.mouse().y + 5, 0.3, 0xff0000, 'w:%d, h:%d', self.rectangle.width, self.rectangle.height)

                        cvui.rect(self.processed_image, self.rectangle.x, self.rectangle.y, self.rectangle.width, self.rectangle.height, 0xff0000)

                if cvui.mouse(cvui.UP):
                    if not dic[name]['data']:
                        pass
                    elif dic[name]['type'] == 'areas' and self.rectangle.x == 0:
                        pass
                    else:
                        if dic[name]['type'] == 'line':
                            dic[name]['data'] = dic[name]['data'] + [cvui.mouse().x, cvui.mouse().y]

                        elif dic[name]['type'] == 'area':
                            dic[name]['data'] = dic[name]['data'] + [self.rectangle.width, self.rectangle.height]
                            self.rectangle = cvui.Rect(0, 0, 0, 0)
                        elif dic[name]['type'] == 'areas':
                            dic[name]['data'][-1] = dic[name]['data'][-1] + [self.rectangle.width, self.rectangle.height]
                            self.rectangle = cvui.Rect(0, 0, 0, 0)

                        print (dic[name]['type'], dic[name]['data'])
                        dic[name]['check'][0] = False


    def ui_build(self):
        self.ui_y = 80

        #self features contain frame number, template and data stored in template
        if self.template is not None:
            if self.current not in self.features.keys():
                self.features[self.current] = copy.deepcopy(self.template)

            # build ui from template
            self.ui_feature(self.features[self.current])
        else:
            cvui.checkbox(self.processed_image, 20, self.ui_y, 'test', [False])

        if self.debug:
            print(self.current)
            print(self.features)
        self.ui_checkmouse(self.features[self.current])

        cvui.update()

    def padded_bounding(self, area, img_dims, pad=1.2):
        """preferrably padded bounding"""
        nparea = np.array(area).reshape(2,2)
        center = nparea.mean(axis=0).astype(int)
        print(center)
        halfside = np.array(np.max(nparea[1] - nparea[0])*pad/2).astype(int)

        #new area shape xyxy
        #img_dims shape yx
        newarea = np.array([center-halfside, center+halfside])
        return np.clip(newarea, [0, 0], img_dims).ravel()

    def xyxy_to_yxyx(self, area):
        return np.flip(np.array(area).reshape(2,2,-1), 1).ravel()

    def xyxy_to_xxyy(self, area):
        return np.swapaxes(np.array(area).reshape(2,2,-1), 0, 1).ravel()

    def xxyy_to_yyxx(self, area):
        return np.flip(np.array(area).reshape(2,2,-1), 0).ravel()

    def xydxdy_to_xyxy(self, area):
        """origin, offset to origin, corner"""
        area = np.array(area).reshape(2,2,-1)
        area[1] = area[1] + area[0]
        return area.ravel()

    def xyxy_to_xydxdy(self, area):
        """origin, corner to origin, offset"""
        area = np.array(area).reshape(2,2,-1)
        area[1] = area[1] - area[0]
        return area.ravel()

    def annotation_to_crop(self, area, img_dims, pad=1.2):
        """
            assuming annotations are in origin(x,y),offest(x,y) format
            crop is in origin(y,x) corner (y,x) format
                square
                padded by a factor
            numpy slicing requires [y:y,x:x,:] -> image[crop[0]:crop[2],crop[1]:crop[3],:]

            return crop dimensions, and new relative annotation
        """
        area = self.xydxdy_to_xyxy(area)
        area = self.xyxy_to_yxyx(area)
        crop = self.padded_bounding(area, img_dims, pad=pad)

        # make annotation relative to crop
        crop_annotation = (area.reshape(2,2) - crop.reshape(2,2)[0]).ravel()
        # put in y,x,dy,dx format
        crop_annotation = list(self.xyxy_to_yxyx(self.xyxy_to_xydxdy(crop_annotation)))
        

        return crop, crop_annotation

    def get_unique_name(self, name, ext):
        """
            this outght to be tied to synset, timestamp hash, traceable file source reference
        """
        idx = 0
        while True:
            fname = '%s_%d%s'%(name, idx, ext)
            if not osp.isfile(fname):
                return fname
            idx += 1

    def export_crop(self, area, name, img_dims, folder, subfolder, atype, frame_index, validate=False):

        def jsonint(o):
            """json barfs at serializing np.int64 type"""
            if isinstance(o, np.int64): 
                return int(o)  
            raise TypeError

        if len(area) != 4:
            print(name, 'cannot be saved, incorrect size', area)
            return 

        # for image cropping turn to absolute image xy values in cv2 order, yx
        crop, new_annotation = self.annotation_to_crop(area, img_dims, pad=1.2)
        
        # np crop [y:y,x:x,:]
        crop_im = self.image[crop[0]:crop[2], crop[1]:crop[3], :].copy()
        isvalid = True
        if validate:
            while True:
                cv2.imshow(MAIN_WIN, crop_im)
                if cv2.waitKey(0) & 0xFF in [ord('q'), ord('y')]:
                    break
                elif cv2.waitKey(0) & 0xFF == ord('n'):
                    isvalid = False
                    break

        # print(crop_im.shape, type(crop_im))
        # print(self.image.shape, type(self.image))
        if isvalid:
            fname = "%s_%s_%d"%(name, osp.basename(folder), frame_index)
            fname = self.get_unique_name(osp.join(subfolder, fname), '.png')

            cv2.imwrite(fname, crop_im)

            newfeature = {'feature':name, 'type':atype, 'data':new_annotation}
            #print(newfeature)
            with open(fname+'.json', 'w') as _fo:
                json.dump(newfeature, _fo, default=jsonint)


    def export_partial(self, annotation_frame, folder=None, frame_index=0, 
                       pyramid=False, sequence=0, resize=False):
        """
            split features within each frame crop and export
            annotation_frame is 
        """

        # check that we have a clean set of features
        if not self.out_features:
            self.export_annotations(folder=folder, images=False, partial_images=False)

        #check if aready open
        cleanvid = False
        if self.vid is None:
            self._open()
            cleanvid = True

        # by default, store in a folder same name as video, next to video
        if folder is None:
            folder = osp.splitext(self.io.file)[0]
            if not osp.isdir(folder):
                os.mkdir(folder)

        img_dims = (self.height, self.width)

        for name in annotation_frame:

            # TODO areas, line, marker
            # pyramids
            subfolder = osp.join(folder, name)
            if not osp.isdir(subfolder):
                os.mkdir(subfolder)

            atype = annotation_frame[name]['type']

            if atype == 'area':
                area = copy.copy(annotation_frame[name]['data'])
                self.export_crop(area, name, img_dims, folder, subfolder, atype, frame_index)

            elif atype == 'areas':
                areas = copy.copy(annotation_frame[name]['data'])
                for area in areas:
                    self.export_crop(area, name, img_dims, folder, subfolder, 'area', frame_index, validate=True)

        if cleanvid:
            self._close()

    def export_images(self, folder=None, partial_images=False):
        """ export annotated images
        """
        # check that we have a clean set of features
        if not self.out_features:
            self.export_annotations(folder=folder, images=False, partial_images=False)

        #check if aready open
        cleanvid = False
        if self.vid is None:
            self._open()
            cleanvid = True

        # by default, store in a folder same name as video, next to video
        if folder is None:
            folder = osp.splitext(self.io.file)[0]
            if not osp.isdir(folder):
                os.mkdir(folder)

        basename = osp.splitext(osp.basename(self.io.file))[0]
        basename = osp.join(folder, basename)

        for frame in self.out_features:
            self._set_frame(frame)
            _, self.image = self.vid.read()

            fname = basename+'.'+str(frame)+'.png'
            # write per frame annotation
            with open(fname+'.json', 'w') as _fo:
                json.dump(self.out_features, _fo)
            # write image
            cv2.imwrite(fname, self.image)

            if partial_images:
                annotation_frame = self.out_features[frame]
                self.export_partial(annotation_frame, folder=folder, frame_index=frame)

        if cleanvid:
            self._close()

    def export_annotations(self, folder=None, fname=None, images=False, partial_images=False):
        """
        export annotations for video
        TODO: ensure compatibility of formats
        """
        self.out_features = {} # clean up just in case
        for frame in self.features:
            self.out_features[frame] = {}
            # print(frame)
            for k in self.features[frame]:
                if self.features[frame][k]['data']:
                    feature = copy.deepcopy(self.features[frame][k])
                    del feature['check']
                    self.out_features[frame][k] = feature

        basename = osp.splitext(self.io.file)[0]
        if fname is None:
            fname = basename+'.json'

        with open(fname, 'w') as _fo:
            json.dump(self.out_features, _fo)

        # export images
        if images:
            self.export_images(folder=folder, partial_images=partial_images)



    def keyhandler(self, key):

        self.current = self.vid.get(cv2.CAP_PROP_POS_FRAMES)

        #if self.debug:

        if key not in (45, 61, 43, ord('q'), 48, ord('m'), ord('d'), 32, 255):
            print(key, '                                       ', end="\r")

        if key == ord('q'):
            print("                                                                    ")
            return False

        #speed control
        if key == 45: # -
            self.wait_ms = min(1000, 2*self.wait_ms)
            print("                                          ", end="\r")
            print("%.3f fps                                "%(1000/self.wait_ms), end="\r")
        elif key in (61, 43): # +
            self.wait_ms = max(1, int(self.wait_ms/2))
            print("                                          ", end="\r")
            print("%.3f fps                                "%(1000/self.wait_ms), end="\r")
        elif key == 48: #0
            self.wait_ms = self.ms_frame
            print("                                          ", end="\r")
            print("original speed %.3f fps                 "%(1000/self.wait_ms), end="\r")

        elif key == ord("l"):
            self._height *= 2
            self._width *= 2
            print("                                          ", end="\r")
            print("resizing window to (%d, %d)                 "%(self._height, self._width), end="\r")
            cv2.resizeWindow(MAIN_WIN, self._width, self._height)
        
        elif key == ord("s"):

            self._height = max(1, int(self._height * 0.5))
            self._width = max(1, int(self._width * 0.5))
            print("                                          ", end="\r")
            print("resizing window to (%d, %d)                 "%(self._height, self._width), end="\r")
            cv2.resizeWindow(MAIN_WIN, self._width, self._height)

        elif key == ord("o"):
            self._height = self.height
            self._width = self.width
            print("                                          ", end="\r")
            print("resizing window to (%d, %d)                 "%(self._height, self._width), end="\r")
            cv2.resizeWindow(MAIN_WIN, self._width, self._height)
            

        # pause: full pause. no read or redraw
        elif key == 32:
            self.pause()

        #markup: no read, continue to redraw
        elif key == ord('m'):
            # only works if theres an annotation file. 
            # unpause spacebar
            self.wait_ms = self.store_ms
            self.ispaused = not self.ispaused
            # if self.ispaused:
            #     self.ui_build()

        # add trackbar
        elif key == ord('d'):
            self.trackbar()
            self.debug = not self.debug

        # elif key == ord('g'):
        #     self.edges = not self.edges
        #     self._prepare_image()

        return True
        #self.wait_ms


    def _set_frame(self, _current):
        self.current = _current
        _ms = self.current * 1000/self.rate
        self.vid.set(cv2.CAP_PROP_POS_MSEC, _ms)


#### Staging
    def _update_play(self, _current=None, reverse=False):

        if _current is not None:
            self._set_frame(_current)

        # LOOP or Bounce TO RESTORE?
        # if self.current > self.last or self.current < self.first:
        #     self._handle_edge()


        # FWD REVERSE TO RESTORE
        # if reverse:
        #     self.direction *= -1

    # replace read vid 
    def _get_image(self, frame=None):

        if frame is not None:
            self._update_play(frame)

        # LOOP TO RESTORE
        # elif self.current > self.last or self.current <= self.first:
        #     self._update_play()

        _, self.image = self.vid.read()
        self._prepare_image()


    def _handle_edge(self):
        '''_handle_edge() toggles between loop and bounce playback'''

        if self.last < self.first:
            _first = self.first
            self.first = self.last
            self.last = _first

        if self.handle_edge > 0: # loop
            self.current = self.first + self.current % (self.last + 1)
            return False

        elif self.handle_edge == 0: # stop
            if self.current >= self.last:
                self._close()
            return False

        elif self.handle_edge < 0: #bounce
            if self.current < self.first:
                self.current = abs(self.current - self.first) + self.first
                self.direction *= -1
            elif self.current > self.last:
                self.current = self.last - self.current % self.last
                self.direction *= -1
        else:
            print("fail?", self.current)

    def _scale(self):
        try:
            _img = cv2.resize(self.image, (0, 0), fx=self.scale, fy=self.scale)
        except:
            _img = self.image
        return _img

    def _advance_frame(self):
        if self.direction < 0:
            self.current -= 2
            self._get_image(self.current)
        else:
            self._get_image()

        # if self.export_imgs:
        #     print(self.current)
        #     self._to_img()

        # if self.export_flow:
        #     self._to_flow()

    def _prepare_image(self):

        self.image_scaled = self._scale()

        if self.edges:

            blur = 7
            cannyth1 = 30
            cannyth2 = 3
            self.image_scaled = cv2.cvtColor(self.image_scaled, cv2.COLOR_BGR2GRAY)
            self.image_scaled = cv2.GaussianBlur(self.image_scaled, (blur, blur), 1.5, 1.5)
            self.image_scaled = cv2.Canny(self.image_scaled, 0, cannyth1, cannyth2)
            self.image_scaled = cv2.bitwise_not(self.image_scaled)


