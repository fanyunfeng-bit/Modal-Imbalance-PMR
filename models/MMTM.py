import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .backbone import resnet18
# import glob
import pickle
# import gin
# from gin.config import _CONFIG

# from src.balanced_mmtm import MMTM_mitigate as MMTM
# from src.balanced_mmtm import get_rescale_weights
from .basic_model import convnet


class MMTM_MVCNN(nn.Module):
    def __init__(self,
                 args,
                 num_views=2,
                 pretraining=False,
                 mmtm_off=False,
                 mmtm_rescale_eval_file_path=None,
                 mmtm_rescale_training_file_path=None,
                 device='cuda:0',
                 saving_mmtm_scales=False,
                 saving_mmtm_squeeze_array=False,
                 ):
        super(MMTM_MVCNN, self).__init__()

        self.args = args
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'CGMNIST':
            n_classes = 10
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.n_classes = n_classes
        self.num_views = num_views
        self.mmtm_off = mmtm_off

        self.saving_mmtm_scales = saving_mmtm_scales
        self.saving_mmtm_squeeze_array = saving_mmtm_squeeze_array

        if args.dataset == 'CREMAD':
            self.net_view_0 = resnet18(modality='audio')
            self.net_view_0_fc = nn.Linear(512, n_classes)
            self.net_view_1 = resnet18(modality='visual')
            self.net_view_1_fc = nn.Linear(512, n_classes)
        elif args.dataset == 'CGMNIST':
            self.net_view_0 = resnet18(modality='audio')
            self.net_view_0_fc = nn.Linear(512, n_classes)
            self.net_view_1 = resnet18(modality='visual')
            self.net_view_1_fc = nn.Linear(512, n_classes)

        self.mmtm2 = MMTM(128, 128, 4)
        self.mmtm3 = MMTM(256, 256, 4)
        self.mmtm4 = MMTM(512, 512, 4)

    def forward(self, audio, visual, curation_mode=False, caring_modality=None):

        frames_view_0 = self.net_view_0.conv1(audio)
        frames_view_0 = self.net_view_0.bn1(frames_view_0)
        frames_view_0 = self.net_view_0.relu(frames_view_0)
        frames_view_0 = self.net_view_0.maxpool(frames_view_0)

        if self.args.dataset == 'CREMAD':
            (B, C, T, H, W) = visual.size()
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            visual = visual.view(B * T, C, H, W)

        frames_view_1 = self.net_view_1.conv1(visual)
        frames_view_1 = self.net_view_1.bn1(frames_view_1)
        frames_view_1 = self.net_view_1.relu(frames_view_1)
        frames_view_1 = self.net_view_1.maxpool(frames_view_1)

        frames_view_0 = self.net_view_0.layer1(frames_view_0)
        frames_view_1 = self.net_view_1.layer1(frames_view_1)

        scales = []
        squeezed_mps = []

        for i in [2, 3, 4]:
            frames_view_0 = getattr(self.net_view_0, f'layer{i}')(frames_view_0)
            frames_view_1 = getattr(self.net_view_1, f'layer{i}')(frames_view_1)

            frames_view_0, frames_view_1, scale, squeezed_mp = getattr(self, f'mmtm{i}')(
                frames_view_0,
                frames_view_1,
                self.saving_mmtm_scales,
                self.saving_mmtm_squeeze_array,
                turnoff_cross_modal_flow=False,
                average_squeezemaps=None,
                curation_mode=curation_mode,
                caring_modality=caring_modality
            )
            scales.append(scale)
            squeezed_mps.append(squeezed_mp)

        # frames_view_0 = self.net_view_0_avgpool(frames_view_0)
        # frames_view_1 = self.net_view_1_avgpool(frames_view_1)

        (_, C, H, W) = frames_view_1.size()
        B = frames_view_0.size()[0]
        frames_view_1 = frames_view_1.view(B, -1, C, H, W)
        frames_view_1 = frames_view_1.permute(0, 2, 1, 3, 4)

        frames_view_0 = F.adaptive_avg_pool2d(frames_view_0, 1)
        frames_view_1 = F.adaptive_avg_pool3d(frames_view_1, 1)

        frames_view_0 = torch.flatten(frames_view_0, 1)
        x_0 = self.net_view_0_fc(frames_view_0)

        frames_view_1 = torch.flatten(frames_view_1, 1)
        x_1 = self.net_view_1_fc(frames_view_1)

        return (x_0 + x_1) / 2, [x_0, x_1], frames_view_0, frames_view_1


class MMTM(nn.Module):
    def __init__(self,
                 dim_visual,
                 dim_skeleton,
                 ratio,
                 device=0,
                 SEonly=False,
                 shareweight=False):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2 * dim / ratio)
        self.SEonly = SEonly
        self.shareweight = shareweight

        self.running_avg_weight_visual = torch.zeros(dim_visual).to("cuda:{}".format(device))
        self.running_avg_weight_skeleton = torch.zeros(dim_visual).to("cuda:{}".format(device))
        self.step = 0

        if self.SEonly:
            self.fc_squeeze_visual = nn.Linear(dim_visual, dim_out)
            self.fc_squeeze_skeleton = nn.Linear(dim_skeleton, dim_out)
        else:
            self.fc_squeeze = nn.Linear(dim, dim_out)

        if self.shareweight:
            assert dim_visual == dim_skeleton
            self.fc_excite = nn.Linear(dim_out, dim_visual)
        else:
            self.fc_visual = nn.Linear(dim_out, dim_visual)
            self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                visual,
                skeleton,
                return_scale=False,
                return_squeezed_mps=False,
                turnoff_cross_modal_flow=False,
                average_squeezemaps=None,
                curation_mode=False,
                caring_modality=0,
                ):

        if self.SEonly:
            tview = visual.view(visual.shape[:2] + (-1,))
            squeeze = torch.mean(tview, dim=-1)
            excitation = self.fc_squeeze_visual(squeeze)
            vis_out = self.fc_visual(self.relu(excitation))

            tview = skeleton.view(skeleton.shape[:2] + (-1,))
            squeeze = torch.mean(tview, dim=-1)
            excitation = self.fc_squeeze_skeleton(squeeze)
            sk_out = self.fc_skeleton(self.relu(excitation))

        else:
            if turnoff_cross_modal_flow:

                tview = visual.view(visual.shape[:2] + (-1,))
                squeeze = torch.cat([torch.mean(tview, dim=-1),
                                     torch.stack(visual.shape[0] * [average_squeezemaps[1]])], 1)
                excitation = self.relu(self.fc_squeeze(squeeze))

                if self.shareweight:
                    vis_out = self.fc_excite(excitation)
                else:
                    vis_out = self.fc_visual(excitation)

                tview = skeleton.view(skeleton.shape[:2] + (-1,))
                squeeze = torch.cat([torch.stack(skeleton.shape[0] * [average_squeezemaps[0]]),
                                     torch.mean(tview, dim=-1)], 1)
                excitation = self.relu(self.fc_squeeze(squeeze))
                if self.shareweight:
                    sk_out = self.fc_excite(excitation)
                else:
                    sk_out = self.fc_skeleton(excitation)

            else:
                squeeze_array = []
                for tensor in [visual, skeleton]:
                    tview = tensor.view(tensor.shape[:2] + (-1,))
                    squeeze_array.append(torch.mean(tview, dim=-1))

                squeeze = torch.cat(squeeze_array, 1)
                excitation = self.fc_squeeze(squeeze)
                excitation = self.relu(excitation)

                if self.shareweight:
                    sk_out = self.fc_excite(excitation)
                    vis_out = self.fc_excite(excitation)
                else:
                    vis_out = self.fc_visual(excitation)
                    sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        self.running_avg_weight_visual = (vis_out.mean(0) + self.running_avg_weight_visual * self.step).detach() / (
                    self.step + 1)
        self.running_avg_weight_skeleton = (vis_out.mean(0) + self.running_avg_weight_skeleton * self.step).detach() / (
                    self.step + 1)

        self.step += 1

        if return_scale:
            scales = [vis_out.cpu(), sk_out.cpu()]
        else:
            scales = None

        if return_squeezed_mps:
            squeeze_array = [x.cpu() for x in squeeze_array]
        else:
            squeeze_array = None

        if not curation_mode:
            dim_diff = len(visual.shape) - len(vis_out.shape)
            vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

            dim_diff = len(skeleton.shape) - len(sk_out.shape)
            sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        else:
            if caring_modality == 0:
                dim_diff = len(skeleton.shape) - len(sk_out.shape)
                sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

                dim_diff = len(visual.shape) - len(vis_out.shape)
                vis_out = torch.stack(vis_out.shape[0] * [
                    self.running_avg_weight_visual
                ]).view(vis_out.shape + (1,) * dim_diff)

            elif caring_modality == 1:
                dim_diff = len(visual.shape) - len(vis_out.shape)
                vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

                dim_diff = len(skeleton.shape) - len(sk_out.shape)
                sk_out = torch.stack(sk_out.shape[0] * [
                    self.running_avg_weight_skeleton
                ]).view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out, skeleton * sk_out, scales, squeeze_array


def get_mmtm_outputs(eval_save_path, mmtm_recorded, key):
    with open(os.path.join(eval_save_path, 'history.pickle'), 'rb') as f:
        his_epo = pickle.load(f)

    print(his_epo.keys())
    data = []
    for batch in his_epo[key][0]:
        assert mmtm_recorded == len(batch)

        for mmtmid in range(len(batch)):
            if len(data) < mmtmid + 1:
                data.append({})
            for i, viewdd in enumerate(batch[mmtmid]):
                data[mmtmid].setdefault('view_%d' % i, []).append(np.array(viewdd))

    for mmtmid in range(len(data)):
        for k, v in data[mmtmid].items():
            data[mmtmid][k] = np.concatenate(data[mmtmid][k])[np.argsort(his_epo['test_indices'][0])]

    return data


def get_rescale_weights(eval_save_path,
                        training_save_path,
                        key='test_squeezedmaps_array_list',
                        validation=False,
                        starting_mmtmindice=1,
                        mmtmpositions=4,
                        device=None,
                        ):
    data = get_mmtm_outputs(eval_save_path, mmtmpositions - starting_mmtmindice, key)

    with open(os.path.join(training_save_path, 'history.pickle'), 'rb') as f:
        his_ori = pickle.load(f)

    selected_indices = his_ori['val_indices'][0] if validation else his_ori['train_indices'][0]

    mmtm_weights = []
    for mmtmid in range(mmtmpositions):
        if mmtmid < starting_mmtmindice:
            mmtm_weights.append(None)
        else:
            weights = [data[mmtmid - starting_mmtmindice][k][selected_indices].mean(0) \
                       for k in sorted(data[mmtmid - starting_mmtmindice].keys())]
            if device is not None:
                weights = numpy_to_torch(weights)
                weights = torch_to(weights, device)
            mmtm_weights.append(weights)

    return mmtm_weights


def numpy_to_torch(obj):
    fn = lambda a: torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    return _apply(obj, fn)


def _apply(obj, func):
    if isinstance(obj, (list, tuple)):
        return type(obj)(_apply(el, func) for el in obj)
    if isinstance(obj, dict):
        return {k: _apply(el, func) for k, el in obj.items()}
    return func(obj)


def torch_to(obj, *args, **kargs):
    return torch_apply(obj, lambda t: t.to(*args, **kargs))


def torch_apply(obj, func):
    fn = lambda t: func(t) if torch.is_tensor(t) else t
    return _apply(obj, fn)