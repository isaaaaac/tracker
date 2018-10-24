# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

from network import trakerCnn 


my_trakerCnn = trakerCnn()
my_trakerCnn.train()
