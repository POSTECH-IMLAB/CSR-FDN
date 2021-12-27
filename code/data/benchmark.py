import os
from . import srdata

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)
        self.args = args

    def _scan(self):
        list_hr = []
        list_lr = []

        for i in range(0, int(len(os.listdir(self.dir_hr)))):
            filename = 'img_{:0>3}_SRF'.format(i+1, self.args.scale[0])
            list_hr.append(os.path.join(self.dir_hr, filename + '_4_HR' + self.ext))
            list_lr.append(os.path.join(self.dir_lr, filename + self.ext))

        list_hr.sort()
        list_lr.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic', 'X{}'.format(self.args.scale[0]))
        self.ext = '.png'
