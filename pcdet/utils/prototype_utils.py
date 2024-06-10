import torch
from torch.functional import F
from torchmetrics import Metric
import numpy as np
from torch.distributions import Categorical
import random

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
        self.pp_feats = None
        self.pp_labels = None

        # Local feature/label which are used to update the global ones
        self.add_state('feats', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')
        self.add_state('ins_ids', default=[], dist_reduce_fx='cat')
        self.add_state('smpl_ids', default=[], dist_reduce_fx='cat')
        self.add_state('iterations', default=[], dist_reduce_fx='cat')
        self.add_state('pl_feats', default=[], dist_reduce_fx='cat')
        self.add_state('pl_labels', default=[], dist_reduce_fx='cat')

    def _init(self, unique_ins_ids, labels):
        self.bank_size = len(unique_ins_ids)
        print(f"Initializing the feature bank with size {self.bank_size} and feature size {self.feat_size}")
        self.prototypes = torch.zeros((self.bank_size, self.feat_size)).cuda()
        self.classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        self.proto_labels = labels
        # self.pp_feats = pl_feats
        # self.pp_labels = pl_labels
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
        
        # if not len(pl_feats)==0:
        #         self.pl_feats.extend(pl_feats)                 # (N, C)
        #         self.pl_labels.extend(pl_labels)      # (N,)
    def compute(self):
        try:
            unique_smpl_ids = torch.unique(torch.cat((self.smpl_ids,), dim=0))
        except:
            unique_smpl_ids = torch.unique(torch.cat((self.smpl_ids), dim=0))
        if len(unique_smpl_ids) < self.reset_state_interval:
            return None

        try:
            features = torch.cat((self.feats,), dim=0)
            ins_ids = torch.cat((self.ins_ids,),dim=0).int().cpu().numpy()
            labels = torch.cat((self.labels,), dim=0).int()
            iterations = torch.cat((self.iterations,),dim=0).int().cpu().numpy()
            ins_ids = torch.cat((self.ins_ids,), dim=0).int().cpu().numpy()
            iterations = torch.cat((self.iterations,), dim=0).int().cpu().numpy()
            # pl_feats = torch.cat((self.pl_feats,),dim=0)
            # pl_labels = torch.cat((self.pl_labels,),dim=0).int()
        except:
            features = torch.cat((self.feats), dim=0)
            ins_ids = torch.cat(self.ins_ids).int().cpu().numpy()
            labels = torch.cat((self.labels), dim=0).int()
            iterations = torch.cat(self.iterations).int().cpu().numpy()
            ins_ids = torch.cat((self.ins_ids), dim=0).int().cpu().numpy()
            iterations = torch.cat((self.iterations), dim=0).int().cpu().numpy()            
            # pl_feats = torch.cat((self.pl_feats),dim=0)
            # pl_labels = torch.cat((self.pl_labels),dim=0).int()

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

    def is_initialized(self):
        return self.initialized
    
    def lpcont_indices(self,nk_labels,mk_labels,k):
        """
        param mk_labels : Sorted labels of pseudo-positives
        return:
        """
        mask = mk_labels == k
        mask_others_denom = mk_labels != k
        mk_first_idx = torch.where(mask)[0][0]
        mk_last_idx = (torch.where(mask)[0][-1]) + 1
        return mask, mask_others_denom, mk_first_idx, mk_last_idx
    
    
    def topk_padding(self,pseudo_positive_labels, pseudo_positives, topk_list,k):
        num_to_append = topk_list[k] - ((torch.nonzero(pseudo_positive_labels==(k+1)).size(0)))
        labels = torch.tensor([(k+1)] * num_to_append).to(pseudo_positives.device)
        pseudo_positive_labels = torch.cat((pseudo_positive_labels, labels), dim=0) 
        pseudo_positives = torch.cat((pseudo_positives, torch.zeros(num_to_append,256).to(pseudo_positives.device)),dim=0) 
        return pseudo_positive_labels, pseudo_positives

    def get_lpcont_loss(self, pseudo_positives, pseudo_positive_labels, topk_list):
        """
        param pseudo_positives: Pseudo positive student features(Mk, Channel=256)
        param topk_labels: Labels for pseudo positive student features
        return:
        """
        N = len(self.prototypes)
        contrastive_loss = torch.tensor(0.0).to(pseudo_positives.device) #contrastive_loss2 = torch.tensor(0.0).to(pseudo_positives.device)
        car_loss = torch.tensor(0.0).to(pseudo_positives.device)
        ped_loss = torch.tensor(0.0).to(pseudo_positives.device)
        cyc_loss = torch.tensor(0.0).to(pseudo_positives.device)
        sorted_labels, sorted_args = torch.sort(self.proto_labels)
        sorted_prototypes = self.prototypes[sorted_args] # sort prototypes to arrange classwise
        
        ori_pseudo_positive_labels = pseudo_positive_labels.clone()
        ori_pseudo_positives = pseudo_positives.clone()

        if torch.nonzero(pseudo_positive_labels==2).shape[0] < topk_list[1]: #padding 5
            pseudo_positive_labels, pseudo_positives = self.topk_padding(pseudo_positive_labels, pseudo_positives, topk_list, k=1)
            # num2_to_append = topk_list[1] - ((torch.nonzero(pseudo_positive_labels==2).size(0)))
            # additional_2s = torch.tensor([2] * num2_to_append).to(pseudo_positives.device)
            # pseudo_positive_labels = torch.cat((pseudo_positive_labels, additional_2s), dim=0)
            # pseudo_positives = torch.cat((pseudo_positives, torch.zeros(num2_to_append,256).to(pseudo_positives.device)),dim=0) #pad classes with zeros until topk satisfied
        
        if torch.nonzero(pseudo_positive_labels==3).shape[0] < topk_list[2]:
            pseudo_positive_labels, pseudo_positives = self.topk_padding(pseudo_positive_labels, pseudo_positives, topk_list, k=2)
            # num3_to_append = topk_list[2] - ((torch.nonzero(pseudo_positive_labels==3).size(0)))
            # additional_3s = torch.tensor([3] * num3_to_append).to(pseudo_positives.device)
            # pseudo_positive_labels = torch.cat((pseudo_positive_labels,additional_3s),dim=0)
            # pseudo_positives = torch.cat((pseudo_positives, torch.zeros(num3_to_append,256).to(pseudo_positives.device)),dim=0) #pad classes with zeros until topk satisfied

        sorted_pp_labels, sorted_pp_args = torch.sort(pseudo_positive_labels)
        sorted_pp_features = pseudo_positives[sorted_pp_args] # sort pseudo positives to arrange classwise  
        padded_features_mask = torch.logical_not(torch.eq(sorted_pp_features, 0).all(dim=-1))

        pp_car_mask = torch.nonzero(sorted_pp_labels==1).view(-1)
        pp_ped_mask = torch.nonzero(sorted_pp_labels==2).view(-1)
        pp_cyc_mask = torch.nonzero(sorted_pp_labels==3).view(-1)
        lbl_car_mask = torch.nonzero(sorted_labels==1).view(-1)
        lbl_car_neg = torch.nonzero(sorted_labels!=1).view(-1)
        lbl_ped_mask = torch.nonzero(sorted_labels==2).view(-1)
        lbl_ped_neg = torch.nonzero(sorted_labels!=2).view(-1)
        lbl_cyc_mask = torch.nonzero(sorted_labels==3).view(-1)
        lbl_cyc_neg = torch.nonzero(sorted_labels!=3).view(-1)
        
        # Index the topk indices, and repeat to match size of Labeled bank instances
        car_mk = torch.topk(pp_car_mask,5)[0]
        ped_mk = torch.topk(pp_ped_mask,5)[0]
        cyc_mk = torch.topk(pp_cyc_mask,5)[0]

        pp_repeated_car = sorted_pp_features[car_mk].repeat(lbl_car_mask.size(0) // car_mk.size(0) + 1, 1)[:lbl_car_mask.size(0)]
        pp_repeated_ped = sorted_pp_features[ped_mk].repeat(lbl_ped_mask.size(0) // ped_mk.size(0) + 1, 1)[:lbl_ped_mask.size(0)]
        pp_repeated_cyc = sorted_pp_features[cyc_mk].repeat(lbl_cyc_mask.size(0) // cyc_mk.size(0) + 1, 1)[:lbl_cyc_mask.size(0)]
        pp_repeated_lbl_car = sorted_pp_labels[car_mk[0]].repeat(pp_repeated_car.size(0))
        pp_repeated_lbl_ped = sorted_pp_labels[ped_mk[0]].repeat(pp_repeated_ped.size(0))
        pp_repeated_lbl_cyc = sorted_pp_labels[cyc_mk[0]].repeat(pp_repeated_cyc.size(0))
            
        pp_fts_repeated_all = torch.cat((pp_repeated_car, pp_repeated_ped, pp_repeated_cyc), dim=0)
        pp_labels_repeated_all = torch.cat((pp_repeated_lbl_car, pp_repeated_lbl_ped, pp_repeated_lbl_cyc), dim=0)

        sorted_prototypes = F.normalize(sorted_prototypes, dim=-1)
        pp_fts_repeated_all = F.normalize(pp_fts_repeated_all, dim=-1)

        padded_features_mask = torch.logical_not(torch.eq(pp_fts_repeated_all, 0).all(dim=-1)) # record padded indices

        sorted_labels = sorted_labels.contiguous().view(-1,1)
        label_mask = (sorted_labels.unsqueeze(1) == pp_labels_repeated_all.unsqueeze(0)).float()
        # label_mask = torch.eq(sorted_labels, sorted_labels.T).float() #label_mask = sorted_labels == ori_pseudo_positive_labels

        sim_pos_pp_matrix = sorted_prototypes @ pp_fts_repeated_all.t()
        # sim_pos_pp_matrix = pp_fts_repeated_all@sorted_prototypes.t()
        
        sim_pos_pp_matrix = torch.exp(sim_pos_pp_matrix/1.0) #division by temperature
        contrastive_loss = torch.log((sim_pos_pp_matrix*(label_mask==1.0))[:,padded_features_mask].sum()/(sim_pos_pp_matrix*(label_mask==0.0))[:,padded_features_mask].sum())
        # for i in range(sim_pos_pp_matrix.size(0)):
        #     denominator = ((sim_pos_pp_matrix[i] * (label_mask[i] == 0.0))[padded_features_mask]).sum(dim=-1)
        #     assert torch.all(denominator > 1e-8), "Zero or near-zero values found in denominator, leading to potential Inf loss."
        #     contrastive_loss = contrastive_loss + torch.log((((sim_pos_pp_matrix[i]*(label_mask[i]==1.0))[padded_features_mask]).sum(dim=-1)) / (((sim_pos_pp_matrix[i]*(label_mask[i]==0.0))[padded_features_mask]).sum(dim=-1)))
                                               
        contrastive_loss = contrastive_loss * -1 / (sorted_prototypes.size(0) * 3 * topk_list[0]) 

        # pp_car_mask = torch.nonzero(pp_labels_repeated_all==1).view(-1)
        # pp_ped_mask = torch.nonzero(pp_labels_repeated_all==2).view(-1)
        # pp_cyc_mask = torch.nonzero(pp_labels_repeated_all==3).view(-1)
        # lbl_car_mask = torch.nonzero(sorted_labels==1).view(-1)
        # lbl_car_neg = torch.nonzero(sorted_labels!=1).view(-1)
        # lbl_ped_mask = torch.nonzero(sorted_labels==2).view(-1)
        # lbl_ped_neg = torch.nonzero(sorted_labels!=2).view(-1)
        # lbl_cyc_mask = torch.nonzero(sorted_labels==3).view(-1)
        # lbl_cyc_neg = torch.nonzero(sorted_labels!=3).view(-1)

        # if pp_car_mask.any():
        #     car_sims = torch.cat((sim_pos_pp_matrix[pp_car_mask][:,lbl_car_mask], sim_pos_pp_matrix[pp_car_mask][:,lbl_car_neg]),dim=1)
        #     car_labels = torch.zeros_like(car_sims)
        #     car_eye = torch.eye(car_labels.size(0), device=car_labels.device)
        #     car_labels[:, :car_eye.size(1)] = car_eye
        #     car_loss = -1 * car_labels * F.log_softmax(car_sims,dim=-1)
        #     contrastive_loss = contrastive_loss + car_loss.sum()
        # if pp_ped_mask.any():
        #     ped_sims = torch.cat((sim_pos_pp_matrix[pp_ped_mask][:,lbl_ped_mask], sim_pos_pp_matrix[pp_ped_mask][:,lbl_ped_neg]),dim=1)
        #     ped_labels = torch.zeros_like(ped_sims)
        #     ped_eye = torch.eye(ped_labels.size(0), device=ped_labels.device)
        #     ped_labels[:, :ped_eye.size(1)] = ped_eye
        #     ped_loss = -1 * ped_labels * F.log_softmax(ped_sims,dim=-1)
        #     contrastive_loss = contrastive_loss + ped_loss.sum()
        # if pp_cyc_mask.any():
        #     cyc_sims = torch.cat((sim_pos_pp_matrix[pp_cyc_mask][:,lbl_cyc_mask], sim_pos_pp_matrix[pp_cyc_mask][:,lbl_cyc_neg]),dim=1)
        #     cyc_labels = torch.zeros_like(cyc_sims)
        #     cyc_eye = torch.eye(cyc_labels.size(0), device=cyc_labels.device)
        #     cyc_labels[:, :cyc_eye.size(1)] = cyc_eye
        #     cyc_loss = -1 * cyc_labels * F.log_softmax(cyc_sims,dim=-1)
        #     contrastive_loss = contrastive_loss + cyc_loss.sum()
            
        # contrastive_loss = contrastive_loss/ (3 * sorted_prototypes.size(0)) # num_classes
        # contrastive_loss = contrastive_loss.sum()/ (sorted_pp_features.size(0)) # num_classes
        return contrastive_loss


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