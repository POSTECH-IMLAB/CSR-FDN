import os
from . import srdata

class vid4(srdata.SRData):
    def __init__(self, args, train=True):
        super(vid4, self).__init__(args, train, benchmark=True)
        self.args = args

    def _scan(self):
        list_hr = []
        list_lr = []

        for dir in self.vid_list:
            vid_path = os.path.join(self.apath, dir)
            hr_path = os.path.join(vid_path, self.hr_dir)
            lr_path = os.path.join(vid_path, self.lr_dir)
            for fname in os.listdir(hr_path):
                #filename = 'img_{:0>3}_SRF'.format(i+1, self.args.scale[0])
                list_hr.append(os.path.join(hr_path, fname))
            for fname in os.listdir(lr_path):
                list_lr.append(os.path.join(lr_path, fname))

        list_hr.sort()
        list_lr.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.vid_list = os.listdir(self.apath)
        self.hr_dir = 'truth'
        self.lr_dir = 'blur4'
        self.ext = '.png'
