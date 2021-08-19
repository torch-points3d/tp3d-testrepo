import numpy as np
from typing import List
import shutil
import matplotlib.pyplot as plt
import os
from os import path as osp
import torch
import logging
from collections import namedtuple
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from .enums import ConvolutionFormat
from torch_points3d.utils.debugging_vars import DEBUGGING_VARS
import subprocess

log = logging.getLogger(__name__)


class ConvolutionFormatFactory:
    @staticmethod
    def check_is_dense_format(conv_type):
        if (
            conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value.lower()
            or conv_type.lower() == ConvolutionFormat.MESSAGE_PASSING.value.lower()
            or conv_type.lower() == ConvolutionFormat.SPARSE.value.lower()
        ):
            return False
        elif conv_type.lower() == ConvolutionFormat.DENSE.value.lower():
            return True
        else:
            raise NotImplementedError("Conv type {} not supported".format(conv_type))
