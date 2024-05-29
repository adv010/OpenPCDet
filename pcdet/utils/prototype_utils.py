import torch
from torch.functional import F
from torchmetrics import Metric
import numpy as np
from torch.distributions import Categorical


class FeatureBank(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):

        super().__init__()
        self.tag = kwargs.get('NAME', None)

        self.temperature = kwargs.get('TEMPERATURE')
        self.num_classes = 3
        self.feat_size = kwargs.get('FEATURE_SIZE')
        self.bank_size = kwargs.get('BANK_SIZE')  # e.g., num. of classes or labeled instances
        self.momentum = kwargs.get('MOMENTUM')
        self.direct_update = kwargs.get('DIRECT_UPDATE')
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL')  # reset the state when N unique samples are seen
        self.num_points_thresh = kwargs.get('FILTER_MIN_POINTS_IN_GT', 0)

        self.initialized = False
        self.insId_protoId_mapping = None  # mapping from instance index to prototype index

        # Globally synchronized prototypes used in each process
        self.prototypes = None
        self.classwise_prototypes = None
        self.classwise_meta_instance = None
        self.proto_labels = None
        self.num_updates = None

        # Local feature/label which are used to update the global ones
        self.add_state('feats', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')
        self.add_state('ins_ids', default=[], dist_reduce_fx='cat')
        self.add_state('smpl_ids', default=[], dist_reduce_fx='cat')
        self.add_state('iterations', default=[], dist_reduce_fx='cat')

    def _init(self, unique_ins_ids, labels):
        self.bank_size = len(unique_ins_ids)
        print(f"Initializing the feature bank with size {self.bank_size} and feature size {self.feat_size}")
        self.prototypes = torch.zeros((self.bank_size, self.feat_size)).cuda()
        self.classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        self.classwise_meta_instance= torch.zeros((3, self.feat_size)).cuda()
        self.proto_labels = labels
        self.num_updates = torch.zeros(self.bank_size).cuda()
        self.insId_protoId_mapping = {unique_ins_ids[i]: i for i in range(len(unique_ins_ids))}

    def update(self, feats: [torch.Tensor], labels: [torch.Tensor], ins_ids: [torch.Tensor], smpl_ids: torch.Tensor,
               iteration: int) -> None:
        for i in range(len(feats)):
            self.feats.append(feats[i])                 # (N, C)
            self.labels.append(labels[i].view(-1))      # (N,)
            self.ins_ids.append(ins_ids[i].view(-1))    # (N,)
            self.smpl_ids.append(smpl_ids[i].view(-1))  # (1,)
            rois_iter = torch.tensor(iteration, device=feats[0].device).expand_as(ins_ids[i].view(-1))
            self.iterations.append(rois_iter)           # (N,)

    def compute(self):
        unique_smpl_ids = torch.unique(torch.cat(self.smpl_ids))
        if len(unique_smpl_ids) < self.reset_state_interval:
            return None

        features = torch.cat(self.feats)
        labels = torch.cat(self.labels).int()
        ins_ids = torch.cat(self.ins_ids).int().cpu().numpy()
        iterations = torch.cat(self.iterations).int().cpu().numpy()
        assert len(features) == len(labels) == len(ins_ids) == len(iterations), \
            "length of features, labels, ins_ids, and iterations should be the same"
        sorted_ins_ids, arg_sorted_ins_ids = np.sort(ins_ids), np.argsort(ins_ids)
        unique_ins_ids, split_indices = np.unique(sorted_ins_ids, return_index=True)

        if not self.initialized:
            self._init(unique_ins_ids, labels[arg_sorted_ins_ids[split_indices]])

        # Group by ins_ids
        inds_groupby_ins_ids = np.split(arg_sorted_ins_ids, split_indices[1:])
        # For each group sort instance ids by iterations in ascending order and apply reduction operation
        for grouped_inds in inds_groupby_ins_ids:
            grouped_inds = grouped_inds[np.argsort(iterations[grouped_inds])]
            ins_id = ins_ids[grouped_inds[0]]
            proto_id = self.insId_protoId_mapping[ins_id]
            assert torch.allclose(labels[grouped_inds[0]], labels[grouped_inds]), "labels should be the same for the same instance id"

            if not self.initialized or self.direct_update:
                self.num_updates[proto_id] += len(grouped_inds)
                new_prototype = torch.mean(features[grouped_inds], dim=0, keepdim=True)  # TODO: maybe it'd be better to replaced it by the EMA
                self.prototypes[proto_id] = new_prototype
            else:
                for ind in grouped_inds:
                    new_prototype = self.momentum * self.prototypes[proto_id] + (1 - self.momentum) * features[ind]
                    self.prototypes[proto_id] = new_prototype
        self._update_classwise_prototypes()
        self._update_classwise_meta_instance()
        self.initialized = True
        self.reset()
        return self.prototypes, self.proto_labels, self.num_updates

    def _update_classwise_prototypes(self):
        classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        for i in range(self.num_classes):  # TODO: refactor it
            inds = torch.where(self.proto_labels == (i+1))[0]
            print(f"Update classwise prototypes for class {(i+1)} with {len(inds)} instances.")
            classwise_prototypes[i] = torch.mean(self.prototypes[inds], dim=0)
        self.classwise_prototypes = self.momentum * self.classwise_prototypes + (1 - self.momentum) * classwise_prototypes

        # prototype[256]
        # prtotp[0]  + prptp[1] + .....

    def _update_classwise_meta_instance(self): # E_{k} per Inf-Mining in CoIN

        classwise_meta_instance = torch.zeros((3, self.feat_size)).cuda()
        for i in range(self.num_classes):  # TODO: refactor it
            inds = torch.where(self.proto_labels == (i+1))[0]
            print(f"Update meta_instances for class {(i+1)}")
            meta_result = (F.normalize(self.prototypes[inds], dim=-1) @ F.normalize(self.classwise_prototypes[i], dim=-1).t())
            pad_length = self.feat_size - meta_result.size(0)
            pad_tensor = torch.zeros((pad_length,) + meta_result.size()[1:]).to(self.prototypes.device)
            padded_result = torch.cat((meta_result,pad_tensor),dim=0)
            classwise_meta_instance[i] = padded_result
        self.classwise_meta_instance = self.momentum * self.classwise_meta_instance + (1 - self.momentum) * classwise_meta_instance

    @torch.no_grad()
    def get_sim_scores(self, input_features, use_classwise_prototypes=True):
        assert input_features.shape[1] == self.feat_size, "input feature size is not equal to the bank feature size"
        if not self.initialized:
            return input_features.new_zeros(input_features.shape[0], 3)
        if use_classwise_prototypes:
            cos_sim = F.normalize(input_features) @ F.normalize(self.classwise_prototypes).t()
            return F.softmax(cos_sim / self.temperature, dim=-1)
        else:
            self._get_sim_scores_with_instance_prototypes(input_features)

    def _get_sim_scores_with_instance_prototypes(self, input_features):
        cos_sim = F.normalize(input_features) @ F.normalize(self.prototypes).t()
        norm_cos_sim = F.softmax(cos_sim / self.temperature, dim=-1)
        classwise_sim = cos_sim.new_zeros(input_features.shape[0], 3)
        lbs = self.proto_labels.expand_as(cos_sim).long()
        classwise_sim.scatter_add_(1, lbs, norm_cos_sim)
        # classwise_sim.scatter_add_(1, lbs, cos_sim)
        # protos_cls_counts = torch.bincount(self.proto_labels).view(1, -1)
        # classwise_sim /= protos_cls_counts  # Note: not probability
        classwise_sim /= classwise_sim.mean(dim=0)
        return classwise_sim

    def get_pairwise_protos_sim_matrix(self):
        sorted_lbs, arg_sorted_lbs = torch.sort(self.proto_labels)
        protos = self.prototypes[arg_sorted_lbs]
        sim_matrix = F.normalize(protos) @ F.normalize(protos).t()

        return sim_matrix.cpu().numpy(), sorted_lbs.cpu().numpy()

    def get_proto_contrastive_loss(self, feats, labels):
        """
        :param feats: pseudo-box features of the strongly augmented unlabeled samples (N, C)
        :param labels: pseudo-labels of the strongly augmented unlabeled samples (N,)
        :return:
        """
        if not self.initialized:
            return None
        sim_scores = F.normalize(feats) @ F.normalize(self.classwise_prototypes).t()
        log_probs = F.log_softmax(sim_scores / self.temperature, dim=-1)
        return -log_probs[torch.arange(len(labels)), labels]

    def get_lpcont_loss(self, pseudo_positives, topk_labels, topk_list):
        """
        :param feats: pseudo-box features of the strongly augmented unlabeled samples (N, C)
        :param labels: pseudo-labels of the strongly augmented unlabeled samples (N,)
        :return:
        """
        if not self.initialized:
            return None
        N = len(self.prototypes)
        sorted_labels, sorted_args = torch.sort(self.proto_labels)
        sorted_prototypes = self.prototypes[sorted_args] # sort prototypes to arrange classwise
        pseudo_positives_meta = []
        pseudo_positive_labels = []
        for i in range(self.num_classes): # adding meta instance
            pseudo_positives_meta.append(pseudo_positives[i])
            pseudo_positive_labels.append(topk_labels[i])
            pseudo_positives_meta.append(self.classwise_meta_instance[i]) # corresponding meta instance
            pseudo_positive_labels.append(topk_labels[i])

        pseudo_positives_meta = torch.stack(pseudo_positives_meta, dim=0)
        sim_lpcont =F.normalize(sorted_prototypes,dim=-1) @ F.normalize(pseudo_positives_meta,dim=-1).t() / 0.07
        sim_lpcont = torch.exp(sim_lpcont) # [N, k*m_k+1]

        lbl_car_protos = sorted_args[sorted_labels==1]
        lbl_ped_protos = sorted_args[sorted_labels==2]
        lbl_cyc_protos = sorted_args[sorted_labels==3]
        car_pseudo = 0
        car_meta = 1
        ped_pseudo = 2
        ped_meta = 3
        cyc_pseudo = 4
        cyc_meta = 5
        loss = 0

        '''
        161 labeled instances  <<----Car ---->> << ----- Ped ------>> << ----- Cyc ----->> : rows
        6 pseudo positives : car_pseudo, car_meta, ped_pseudo, ped_meta, cyc_pseudo, cyc_meta : columns
        '''
        for idx_car in lbl_car_protos:
            loss+=torch.log(sim_lpcont[idx_car,car_pseudo]/(sim_lpcont[idx_car,ped_pseudo] + sim_lpcont[idx_car, cyc_pseudo])) 
            loss+=torch.log(sim_lpcont[idx_car,car_meta]/(sim_lpcont[idx_car,ped_meta] + sim_lpcont[idx_car, cyc_meta]))
        for idx_ped in lbl_ped_protos:
            loss+=torch.log(sim_lpcont[idx_ped,ped_pseudo]/sim_lpcont[idx_ped,car_pseudo] + sim_lpcont[idx_ped, cyc_pseudo])
            loss+=torch.log(sim_lpcont[idx_ped,ped_meta]/sim_lpcont[idx_ped,car_meta] + sim_lpcont[idx_ped, cyc_meta])
        for idx_cyc in lbl_cyc_protos:
            loss+=torch.log(sim_lpcont[idx_cyc,cyc_pseudo]/sim_lpcont[idx_cyc,car_pseudo] + sim_lpcont[idx_cyc, ped_pseudo])
            loss+=torch.log(sim_lpcont[idx_cyc,cyc_meta]/sim_lpcont[idx_cyc,car_meta] + sim_lpcont[idx_cyc, ped_meta])
        
        constant = -1 / (self.bank_size * 2 * self.num_classes) # m_k+1 = 2
        loss = constant * loss
        return loss

class FeatureBankRegistry(object):
    def __init__(self, **kwargs):
        self._banks = {}

    def register(self, tag=None, **bank_configs):
        if tag is None:
            tag = 'default'
        if tag in self.tags():
            raise ValueError(f'Feature bank with tag {tag} already exists')
        bank = FeatureBank(**bank_configs)
        self._banks[tag] = bank
        return self._banks[tag]

    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag not in self.tags():
            raise ValueError(f'Feature bank with tag {tag} does not exist')
        return self._banks[tag]

    def tags(self):
        return self._banks.keys()


feature_bank_registry = FeatureBankRegistry()
