import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torchvision.models as models
import torch.nn.functional as F
import time
import librosa
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from torch.quantization import QuantStub, DeQuantStub

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tfeb_pool_size_component(length):
    # print(length);
    c = [];
    index = 1;
    while index <= 6:
        if length >= 2:
            if index == 6:
                c.append(length);
            else:
                c.append(2);
                length = length // 2;
        else:
           c.append(1);
        index += 1;
    return c

def get_tfeb_pool_sizes(conv2_ch, width):
    h = get_tfeb_pool_size_component(conv2_ch);
    w = get_tfeb_pool_size_component(width);
    # print(w);
    pool_size = [];
    for  (h1, w1) in zip(h, w):
        pool_size.append((h1, w1));
    return pool_size

class ElephantCallerNet(nn.Module):
    def __init__(self, input_length, n_class, sr, ch_conf=None, quantize=False):
        super(ElephantCallerNet, self).__init__()
        self.input_length = input_length
        self.ch_config = ch_conf
        self.quantize = quantize

        stride1 = 2
        stride2 = 2
        channels = 8
        k_size = (3, 3)
        n_frames = (sr / 1000) * 10  # No of frames per 10ms

        sfeb_pool_size = int(n_frames / (stride1 * stride2))
        if self.ch_config is None:
            self.ch_config = [channels, channels * 8, channels * 4, channels * 8, channels * 8,
                              channels * 16, channels * 16, channels * 32, channels * 32, channels * 64,
                              channels * 64, n_class]

        fcn_no_of_inputs = self.ch_config[-1]

        self.conv1, self.bn1 = self.make_layers(1, self.ch_config[0], (1, 9), (1, stride1))
        self.conv2, self.bn2 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2))
        self.conv3, self.bn3 = self.make_layers(1, self.ch_config[2], k_size, padding=1)
        self.conv4, self.bn4 = self.make_layers(self.ch_config[2], self.ch_config[3], k_size, padding=1)
        self.conv5, self.bn5 = self.make_layers(self.ch_config[3], self.ch_config[4], k_size, padding=1)
        self.conv6, self.bn6 = self.make_layers(self.ch_config[4], self.ch_config[5], k_size, padding=1)
        self.conv7, self.bn7 = self.make_layers(self.ch_config[5], self.ch_config[6], k_size, padding=1)
        self.conv8, self.bn8 = self.make_layers(self.ch_config[6], self.ch_config[7], k_size, padding=1)
        self.conv9, self.bn9 = self.make_layers(self.ch_config[7], self.ch_config[8], k_size, padding=1)
        self.conv10, self.bn10 = self.make_layers(self.ch_config[8], self.ch_config[9], k_size, padding=1)
        self.conv11, self.bn11 = self.make_layers(self.ch_config[9], self.ch_config[10], k_size, padding=1)
        self.conv12, self.bn12 = self.make_layers(self.ch_config[10], self.ch_config[11], (1, 1))

        self.fcn = nn.Linear(fcn_no_of_inputs, n_class)
        nn.init.kaiming_normal_(self.fcn.weight, nonlinearity='sigmoid')

        self.sfeb = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(),
            self.conv2, self.bn2, nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        )

        tfeb_modules = []
        self.tfeb_width = int(((self.input_length / sr) * 1000) / 10)  # 10ms frames of audio length in seconds
        tfeb_pool_sizes = get_tfeb_pool_sizes(self.ch_config[1], self.tfeb_width)
        p_index = 0
        for i in [3, 4, 6, 8, 10]:
            tfeb_modules.extend([eval('self.conv{}'.format(i)), eval('self.bn{}'.format(i)), nn.ReLU()])

            if i != 3:
                tfeb_modules.extend([eval('self.conv{}'.format(i + 1)), eval('self.bn{}'.format(i + 1)), nn.ReLU()])

            h, w = tfeb_pool_sizes[p_index]
            if h > 1 or w > 1:
                tfeb_modules.append(nn.MaxPool2d(kernel_size=(h, w)))
            p_index += 1

        tfeb_modules.append(nn.Dropout(0.2))
        tfeb_modules.extend([self.conv12, self.bn12, nn.ReLU()])
        h, w = tfeb_pool_sizes[-1]
        if h > 1 or w > 1:
            tfeb_modules.append(nn.AvgPool2d(kernel_size=(h, w)))
        tfeb_modules.extend([nn.Flatten(), self.fcn])

        self.tfeb = nn.Sequential(*tfeb_modules)

        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        )

        if self.quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        if self.quantize:
            x = self.quant(x)

        x = self.sfeb(x)
        x = x.permute((2, 0, 1, 3))
        x = self.tfeb(x)

        if self.quantize:
            x = self.dequant(x)
        y = self.output(x)
        y = F.softmax(y, dim=1)
        return y

    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=bias)
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        bn = nn.BatchNorm2d(out_channels)
        return conv, bn

class Residual_block(nn.Module):
	def __init__(self, nb_filts, first = False):
		super(Residual_block, self).__init__()
		self.first = first
		
		if not self.first:
			self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
		self.lrelu = nn.LeakyReLU()
		self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

		self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
		self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
		self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)

		if nb_filts[0] != nb_filts[1]:
			self.downsample = True
			self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
		else:
			self.downsample = False
		self.mp = nn.MaxPool1d(3)

	def forward(self, x):
		identity = x
		if not self.first:
			out = self.bn1(x)
			out = self.lrelu_keras(out)
		else:
			out = x

		out = self.conv1(x)
		out = self.bn2(out)
		out = self.lrelu_keras(out)
		out = self.conv2(out)

		if self.downsample:
			identity = self.conv_downsample(identity)
		
		out += identity
		out = self.mp(out)
		return out

class RawNet(nn.Module):
	def __init__(self):
		super(RawNet, self).__init__()
		#self.negative_k = d_args['negative_k']
		self.first_conv = nn.Conv1d(in_channels = 1,
			out_channels = 128,
			kernel_size = 3,
			padding = 0,
			stride = 3)
		self.first_bn = nn.BatchNorm1d(num_features = 128)
		self.lrelu = nn.LeakyReLU()
		self.lrelu_keras = nn.LeakyReLU(negative_slope = 0.3)

		self.block0 = self._make_layer(nb_blocks = 2,
			nb_filts = [128,128],
			first = True)
		self.block1 = self._make_layer(nb_blocks = 4,
			nb_filts = [128, 256])

		self.bn_before_gru = nn.BatchNorm1d(num_features = 256)
		self.gru = nn.GRU(input_size = 256,
			hidden_size = 1024,
			num_layers = 1,
			batch_first = True)
		self.fc1_gru = nn.Linear(in_features = 1024,
			out_features = 1024)
		self.fc2_gru = nn.Linear(in_features =1024,
			out_features = 3,
			bias = True)

	def forward(self, x, y = 0, is_test=False):
		# print(x.shape)
		x = x.unsqueeze(1)
		# print(x.shape)
		x = self.first_conv(x)
		x = self.first_bn(x)
		x = self.lrelu_keras(x)

		x = self.block0(x)
		x = self.block1(x)

		x = self.bn_before_gru(x)
		x = self.lrelu_keras(x)
		x = x.permute(0, 2, 1)#(batch, filt, time) >> (batch, time, filt)
		x, _ = self.gru(x)
		x = x[:,-1,:]
		code = self.fc1_gru(x)
		if is_test: return code

		code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
		code = torch.div(code, code_norm)
		out = self.fc2_gru(code)
		return out
		'''
		#for future updates, bc_loss, h_loss
		#h_loss
		norm = torch.norm(self.fc2_gru.weight, dim = 1, keepdim = True)
		normed_weight = torch.div(self.fc2_gru.weight, norm)
		cos_output_tmp = torch.mm(code, normed_weight.t())
		cos_impo = cos_output_tmp.gather(1, y2)
		cos_target = cos_output_tmp.gather(1, y.view(-1, 1))
		hard_negatives, _ = torch.topk(cos_impo, self.negative_k, dim=1, sorted=False)
		hard_negatives = F.relu(hard_negatives, inplace=True)
		trg_score = cos_target*-1.
		h_loss = torch.log(1.+torch.exp(hard_negatives+trg_score).sum(dim=1))
		return out, h_loss
		'''

	def _make_layer(self, nb_blocks, nb_filts, first = False):
		layers = []
		#def __init__(self, nb_filts, first = False):
		for i in range(nb_blocks):
			first = first if i == 0 else False
			layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
			if i == 0: nb_filts[0] = nb_filts[1]

		return nn.Sequential(*layers)

	def summary(self, input_size, batch_size=-1, device="cuda", print_fn = None):
		if print_fn == None: printfn = print
		model = self
	
		def register_hook(module):
	
			def hook(module, input, output):
				class_name = str(module.__class__).split(".")[-1].split("'")[0]
				module_idx = len(summary)
	
				m_key = "%s-%i" % (class_name, module_idx + 1)
				summary[m_key] = OrderedDict()
				summary[m_key]["input_shape"] = list(input[0].size())
				summary[m_key]["input_shape"][0] = batch_size
				if isinstance(output, (list, tuple)):
					summary[m_key]["output_shape"] = [
						[-1] + list(o.size())[1:] for o in output
					]
				else:
					summary[m_key]["output_shape"] = list(output.size())
					if len(summary[m_key]["output_shape"]) != 0:
						summary[m_key]["output_shape"][0] = batch_size
	
				params = 0
				if hasattr(module, "weight") and hasattr(module.weight, "size"):
					params += torch.prod(torch.LongTensor(list(module.weight.size())))
					summary[m_key]["trainable"] = module.weight.requires_grad
				if hasattr(module, "bias") and hasattr(module.bias, "size"):
					params += torch.prod(torch.LongTensor(list(module.bias.size())))
				summary[m_key]["nb_params"] = params
	
			if (
				not isinstance(module, nn.Sequential)
				and not isinstance(module, nn.ModuleList)
				and not (module == model)
			):
				hooks.append(module.register_forward_hook(hook))
	
		device = device.lower()
		assert device in [
			"cuda",
			"cpu",
		], "Input device is not valid, please specify 'cuda' or 'cpu'"
	
		if device == "cuda" and torch.cuda.is_available():
			dtype = torch.cuda.FloatTensor
		else:
			dtype = torch.FloatTensor
		if isinstance(input_size, tuple):
			input_size = [input_size]
		x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
		summary = OrderedDict()
		hooks = []
		model.apply(register_hook)
		model(*x)
		for h in hooks:
			h.remove()
	
		print_fn("----------------------------------------------------------------")
		line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
		print_fn(line_new)
		print_fn("================================================================")
		total_params = 0
		total_output = 0
		trainable_params = 0
		for layer in summary:
			# input_shape, output_shape, trainable, nb_params
			line_new = "{:>20}  {:>25} {:>15}".format(
				layer,
				str(summary[layer]["output_shape"]),
				"{0:,}".format(summary[layer]["nb_params"]),
			)
			total_params += summary[layer]["nb_params"]
			total_output += np.prod(summary[layer]["output_shape"])
			if "trainable" in summary[layer]:
				if summary[layer]["trainable"] == True:
					trainable_params += summary[layer]["nb_params"]
			print_fn(line_new)
	
		# assume 4 bytes/number (float on cuda).
		total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
		total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
		total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
		total_size = total_params_size + total_output_size + total_input_size
	
		print_fn("================================================================")
		print_fn("Total params: {0:,}".format(total_params))
		print_fn("Trainable params: {0:,}".format(trainable_params))
		print_fn("Non-trainable params: {0:,}".format(total_params - trainable_params))
		print_fn("----------------------------------------------------------------")
		print_fn("Input size (MB): %0.2f" % total_input_size)
		print_fn("Forward/backward pass size (MB): %0.2f" % total_output_size)
		print_fn("Params size (MB): %0.2f" % total_params_size)
		print_fn("Estimated Total Size (MB): %0.2f" % total_size)
		print_fn("----------------------------------------------------------------")
		return
	
class MobileNetV2RawAudio(nn.Module):
    def __init__(self, num_classes, num_samples, dropout_rate=0.0, activation='ReLU'):
        super(MobileNetV2RawAudio, self).__init__()
        # Load MobileNetV2 model without the fully connected layer
        self.num_classes=num_classes
        self.num_samples = num_samples

        self.features = models.mobilenet_v2(pretrained=True).features
        # self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=num_classes)
        # # Replace the first convolutional layer to accept 1-channel input
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,2), padding=(1, 1), bias=False)
        
        # # Modify the classifier for the new input shape
        # num_features = self.model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, num_classes),
            self.get_activation(activation) 
        )

    def forward(self, x):
        # if self.num_channels == 1:
        #     x = x.unsqueeze(1)
        # Convert 1-channel input to 3-channel input expected by MobileNetV2
        # x = torch.cat([x, x, x], dim=1)
        x = x.view(-1, 1, self.num_samples // 60, 60)
        
        # Apply the feature extractor layers
        x = self.features(x)
        
        # Apply global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Apply the classifier layers
        x = self.classifier(x)
        
        return x

    def get_activation(self, activation):
        if activation == 'ReLU':
            return nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            return nn.LeakyReLU(inplace=True)
        elif activation == 'Sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function. Please choose from 'ReLU', 'LeakyReLU', or 'Sigmoid'.")
		
class YAMNet(nn.Module):
    def __init__(self, num_classes, num_samples, kernel_size):
        super(YAMNet, self).__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples

        # Define the layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm1d(32, eps=1e-4),
            nn.LeakyReLU(inplace=True)
        )
        self.separable_conv1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, groups=32),
            # nn.BatchNorm1d(32, eps=1e-4),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64, eps=1e-4),
            nn.LeakyReLU(inplace=True)
        )
        self.separable_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, groups=64),
            # nn.BatchNorm1d(64, eps=1e-4),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128, eps=1e-4),
            nn.LeakyReLU(inplace=True)
        )

        self.separable_conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128, eps=1e-4),
            nn.LeakyReLU(inplace=True)
        )
        self.separable_conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, groups=128),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256, eps=1e-4),
            nn.LeakyReLU(inplace=True)
        )

        self.separable_conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, groups=256),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256, eps=1e-4),
            nn.LeakyReLU(inplace=True)
        )
        self.separable_conv6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1, groups=256),
            # nn.BatchNorm1d(256),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.separable_conv7 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, groups=512),
            # nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.separable_conv8 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, groups=512),
            # nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.separable_conv9 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, groups=512),
            # nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.separable_conv10 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1, groups=512),
            nn.Conv1d(1024, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True)
        )
        self.separable_conv11 = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, stride=2, padding=1, groups=1024),
            nn.Conv1d(1024, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(1024, num_classes)

        self.apply(init_weights_he)

    def forward(self, x):
        # print(x.shape)
        # x = x.view(-1, 1, self.num_samples)
        x=x.unsqueeze(1)
        # print("after", x.shape)

        # Apply the convolutional layers
        x = F.relu(self.conv1(x))
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.separable_conv3(x)
        x = self.separable_conv4(x)
        x = self.separable_conv5(x)
        x = self.separable_conv6(x)
        x = self.separable_conv7(x)
        x = self.separable_conv8(x)
        x = self.separable_conv9(x)
        x = self.separable_conv10(x)
        x = self.separable_conv11(x)
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        return x