import io
import os
from collections import defaultdict
from configparser import ConfigParser

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import (Add, BatchNormalization, Concatenate, Conv2D, Input,
                          Lambda, LeakyReLU, MaxPooling2D, UpSampling2D,
                          ZeroPadding2D)
from keras.regularizers import l2


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def load_darknet_weights(model, fp):
    with open(fp, 'rb') as weights_data:
        major, minor, revision, seen, _ = np.fromfile(
            weights_data, dtype=np.int32, count=5
        )
        model_layers = [layer for layer in model.layers if 'lambda' not in layer.name]
        output_layers = [layer for layer in model.layers if 'lambda' in layer.name]
        model_layers.sort(key=lambda layer: layer.name.split('_')[-1])
        model_layers.extend(output_layers)
        for i, layer in enumerate(model_layers):
            current_read = weights_data.tell()
            total_size = os.fstat(weights_data.fileno()).st_size
            print(
                f'\r{round(100 * (current_read / total_size))}'
                f'%\t{current_read}/{total_size}',
                end='',
            )
            if current_read == total_size:
                print()
                break
            if 'conv2d' not in layer.name.lower():
                continue
            next_layer = model_layers[i + 1]
            b_norm_layer = (
                next_layer if 'batch_normalization' in next_layer.name else None
            )
            filters = layer.filters
            kernel_size = layer.kernel_size[0]
            input_dimension = layer.get_input_shape_at(-1)[-1]
            convolution_bias = (
                np.fromfile(weights_data, dtype=np.float32, count=filters)
                if b_norm_layer is None
                else None
            )
            bn_weights = (
                np.fromfile(weights_data, dtype=np.float32, count=4 * filters).reshape(
                    (4, filters)
                )[[1, 0, 2, 3]]
                if (b_norm_layer is not None)
                else None
            )
            convolution_shape = (
                filters,
                input_dimension,
                kernel_size,
                kernel_size,
            )
            convolution_weights = (
                np.fromfile(
                    weights_data,
                    dtype=np.float32,
                    count=np.product(convolution_shape),
                )
                .reshape(convolution_shape)
                .transpose([2, 3, 1, 0])
            )
            if b_norm_layer is None:
                try:
                    layer.set_weights([convolution_weights, convolution_bias])
                except ValueError:
                    pass
            if b_norm_layer is not None:
                layer.set_weights([convolution_weights])
                b_norm_layer.set_weights(bn_weights)
        assert len(weights_data.read()) == 0, 'failed to read all data'


class YoloParser:
    def __init__(
        self,
        total_classes,
    ):
        self.total_classes = total_classes
        self.model_layers = []
        self.previous_layer = None
        self.cfg_parser = None
        self.total_layers = 0

    def create_cfg_parser(self, fp):
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        for line in open(fp):
            if line.startswith('['):
                section = line.strip(' []\n')
                adjusted_section = f'{section}_{section_counters[section]}'
                section_counters[section] += 1
                line = line.replace(section, adjusted_section)
            output_stream.write(line)
        output_stream.seek(0)
        self.cfg_parser = ConfigParser()
        self.cfg_parser.read_file(output_stream)

    def parse_value(self, section, name):
        value = self.cfg_parser[section][name]
        if value.strip('+-').isnumeric():
            return int(value)
        return value

    def create_convolution(self, section):
        filters = self.parse_value(section, 'filters')
        kernel_size = self.parse_value(section, 'size')
        stride = self.parse_value(section, 'stride')
        pad = self.parse_value(section, 'pad')
        activation = self.parse_value(section, 'activation')
        batch_normalize = 'batch_normalize' in self.cfg_parser[section]
        padding = 'same' if pad == 1 and stride == 1 else 'valid'
        if filters == 255:
            filters = 3 * (self.total_classes + 5)
        if stride > 1:
            self.previous_layer = ZeroPadding2D(
                ((1, 0), (1, 0)), name=f'zero_padding2d_{self.total_layers}'
            )(self.previous_layer)
        self.previous_layer = Conv2D(
            filters,
            kernel_size,
            (stride, stride),
            padding,
            use_bias=not batch_normalize,
            kernel_regularizer=l2(0.0005),
            name=f'conv2d_{self.total_layers}',
        )(self.previous_layer)
        if batch_normalize:
            self.previous_layer = BatchNormalization(
                name=f'batch_normalization_{self.total_layers}'
            )(self.previous_layer)
        if activation == 'leaky':
            self.previous_layer = LeakyReLU(
                0.1, name=f'leaky_relu_{self.total_layers}'
            )(self.previous_layer)
        elif activation == 'mish':
            self.previous_layer = mish(self.previous_layer)

    def create_route(self, section):
        layers = [
            self.model_layers[int(i)]
            for i in self.cfg_parser[section]['layers'].split(',')
        ]
        if len(layers) > 1:
            self.previous_layer = Concatenate(name=f'concat_{self.total_layers}')(
                layers
            )
        else:
            self.previous_layer = layers[0]

    def create_max_pool(self, section):
        size = self.parse_value(section, 'size')
        stride = self.parse_value(section, 'stride')
        self.previous_layer = MaxPooling2D(
            (size, size),
            (stride, stride),
            'same',
            name=f'max_pooling2d_{self.total_layers}',
        )(self.previous_layer)

    def create_shortcut(self, section):
        self.previous_layer = Add(name=f'add_{self.total_layers}')(
            [self.model_layers[self.parse_value(section, 'from')], self.previous_layer]
        )

    def create_upsample(self, section):
        stride = self.parse_value(section, 'stride')
        self.previous_layer = UpSampling2D(
            stride, name=f'up_sampling2d_{self.total_layers}'
        )(self.previous_layer)

    def create_output_layer(self):
        self.previous_layer = Lambda(
            lambda x: tf.reshape(x, (-1, *x.shape[1:3], 3, self.total_classes + 5)),
            name=f'lambda_{self.total_layers}',
        )(self.previous_layer)

    def create_section(self, section):
        match section.split('_')[0]:
            case 'convolutional':
                self.create_convolution(section)
            case 'route':
                self.create_route(section)
            case 'maxpool':
                self.create_max_pool(section)
            case 'shortcut':
                self.create_shortcut(section)
            case 'upsample':
                self.create_upsample(section)
            case 'yolo':
                self.create_output_layer()
        self.total_layers += 1
        self.model_layers.append(self.previous_layer)

    def from_cfg(
        self,
        fp,
        input_shape,
    ):
        self.previous_layer = x0 = Input(input_shape)
        self.create_cfg_parser(fp)
        for section in self.cfg_parser.sections():
            self.create_section(section)
        output_layers = [layer for layer in self.model_layers if 'lambda' in layer.name]
        return Model(x0, output_layers)
