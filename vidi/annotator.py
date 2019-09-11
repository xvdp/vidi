import os
import os.path as osp
import json


class Annotator:
    """
        Examples
        # init annotator
        >>> fname = '/home/z/work/pneumonia/extra.json'
        >>> from vidi import Annotator
        >>> a = Annotator()

        # creates annotation template
        >>> a.marker('L4Root')
        ...
        >>> a.line('Spine')
        >>> a.write(fname)
        >>> print(a)

        # creates annotation template from array
        >>> a.annots(["L4Root", "L4Arc", "L9Root", "R4Root", "R4Arc", "R9Root"], "marker")
        >>> a.annots(["Ball","Goal"], "area")
        >>> a.annots(["Mid", "End", "Axis"], "line")

        # loads annotation template
        >>> a.debug=True
        >>> a.load(fname)
    """
    def __init__(self):
        self.A = {}
        self.index = 0
        self.debug = False

    def annot(self, name, a_type):
        assert name not in self.A.keys(), name+' already exists, choose new nmae'
        self.A[name] = {'type':a_type, 'data':[], 'check':[False]}

    def marker(self, name=None):
        self.annot(name, 'marker')
    def area(self, name=None):
        self.annot(name, 'area')
    def line(self, name=None):
        self.annot(name, 'line')

    def annots(self, names, a_type):
        assert isinstance(names, list) and names, 'Annotator().annots(list <names>, str <a_type>)'
        for name in names:
            self.annot(name, a_type)

    def write(self, path='annotation.json'):
        with open(path, 'w') as _fo:
            json.dump(self.A, _fo)

    def load(self, path):
        assert osp.exists(path), path+' not a valid file'
        with open(path, 'r') as _fi:
            self.A = {**json.load(_fi)}
            if self.debug:
                print(self.A)
