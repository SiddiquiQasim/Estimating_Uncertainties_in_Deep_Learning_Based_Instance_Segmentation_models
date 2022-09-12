import numpy as np

from csbdeep.utils import _raise
from csbdeep.utils.tf import keras_import


Input, Conv2D, MaxPooling2D = keras_import('layers', 'Input', 'Conv2D', 'MaxPooling2D')
Dropout = keras_import('layers', 'Dropout')
Model = keras_import('models', 'Model')

from stardist.models import Config2D, StarDist2D
from uncertainty_calculation.addon.uNet import unet_block

class StarDist2D_Unc(StarDist2D):

    def __init__(self, config=Config2D(), name=None, basedir='.', dropout_rate=0.0, mcd_pos='Output'):
        """See class docstring."""
        self.dropout_rate = dropout_rate
        self.mcd_pos = mcd_pos
        super().__init__(config, name=name, basedir=basedir)


    def _build(self):
        self.config.backbone == 'unet' or _raise(NotImplementedError())
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}

        input_img = Input(self.config.net_input_shape, name='input')

        # maxpool input image to grid size
        pooled = np.array([1,1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv2D(self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                    padding='same', activation=self.config.unet_activation)(pooled_img)
            pooled_img = MaxPooling2D(pool)(pooled_img)

        unet_base = unet_block(mcd_pos=self.mcd_pos, drop_rate=self.dropout_rate, **unet_kwargs)(pooled_img)
        ######## ADDED CODE ########
        if self.mcd_pos =='Output' or self.mcd_pos =='Full':
            unet_base = Dropout(self.dropout_rate)(unet_base, training = True)
        ######## END ############

        if self.config.net_conv_after_unet > 0:
            unet = Conv2D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                          name='features', padding='same', activation=self.config.unet_activation)(unet_base)
        else:
            unet = unet_base

        output_prob = Conv2D(                 1, (1,1), name='prob', padding='same', activation='sigmoid')(unet)
        output_dist = Conv2D(self.config.n_rays, (1,1), name='dist', padding='same', activation='linear')(unet)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_unet > 0:
                unet_class  = Conv2D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                                     name='features_class', padding='same', activation=self.config.unet_activation)(unet_base)
            else:
                unet_class  = unet_base
            output_prob_class  = Conv2D(self.config.n_classes+1, (1,1), name='prob_class', padding='same', activation='softmax')(unet_class)
            return Model([input_img], [output_prob,output_dist,output_prob_class])
        else:
            return Model([input_img], [output_prob,output_dist])


    