import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Roi(nn.Module):

    def __init__(self, max_instances=320, max_pooling=True, pooling_size=21):
        super(Roi,self).__init__()
        self.max_instances = max_instances
        self.max_pooling = max_pooling
        self.pooling_size = pooling_size
        self.pre_pool_size = self.pooling_size * 2 if max_pooling else self.pooling_size


    def forward(self, feature_maps:Variable, proposals:Variable, original_size):
        batch_size = len(proposals)
        height, width = feature_maps.shape[2:]
        orig_height, orig_width = original_size
        scales = proposals.data.new([[[width/orig_width, height/orig_height, width/orig_width, height/orig_height]]])
        scales = Variable(scales, requires_grad=False)
        proposals = proposals * scales
        rois = []
        for b in range(batch_size):
            rois.append(self._crop_pool(feature_maps[b],proposals[b]))
        return rois


    def _crop_pool(self, fm, rois):
        """
        [  x2-x1             x1 + x2 - W + 1  ]
        [  -----      0      ---------------  ]
        [  W - 1                  W - 1       ]
        [                                     ]
        [           y2-y1    y1 + y2 - H + 1  ]
        [    0      -----    ---------------  ]
        [           H - 1         H - 1      ]
        """
        x1 = rois[:, 0]
        y1 = rois[:, 1]
        x2 = rois[:, 2]
        y2 = rois[:, 3]

        height = fm.size(1)
        width = fm.size(2)

        # affine theta
        theta = Variable(rois.data.new(rois.size(0), 2, 3).zero_())
        theta[:, 0, 0] = ((x2 - x1) / (width - 1)).view(-1)
        theta[:, 0 ,2] = ((x1 + x2 - width + 1) / (width - 1)).view(-1)
        theta[:, 1, 1] = ((y2 - y1) / (height - 1)).view(-1)
        theta[:, 1, 2] = ((y1 + y2 - height + 1) / (height - 1)).view(-1)

        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, self.pre_pool_size, self.pre_pool_size)))
        crops = F.grid_sample(fm.unsqueeze(0).expand(rois.size(0), -1, -1, -1), grid)
        if self.max_pooling:
            crops = F.max_pool2d(crops, 2, 2)
        return crops
