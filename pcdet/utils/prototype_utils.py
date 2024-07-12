import torch
from torch.functional import F
from torchmetrics import Metric
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.distributed as dist
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
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
        self.initial_temperature = 1.0
        self.final_temperature = 0.5
        self.initial_weight = 0.01
        self.final_weight = 1.0
        self.epochs=60
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
        self.proto_conf_scores = None

        # Local feature/label which are used to update the global ones
        self.add_state('feats', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')
        self.add_state('ins_ids', default=[], dist_reduce_fx='cat')
        self.add_state('smpl_ids', default=[], dist_reduce_fx='cat')
        self.add_state('iterations', default=[], dist_reduce_fx='cat')
        self.add_state('conf_scores', default=[], dist_reduce_fx='cat')

    def _init(self, unique_ins_ids, labels):
        self.bank_size = len(unique_ins_ids)
        print(f"Initializing the feature bank with size {self.bank_size} and feature size {self.feat_size}")
        self.prototypes = torch.zeros((self.bank_size, self.feat_size)).cuda()
        self.classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        self.proto_labels = labels
        self.num_updates = torch.zeros(self.bank_size).cuda()
        self.insId_protoId_mapping = {unique_ins_ids[i]: i for i in range(len(unique_ins_ids))}

    def update(self, feats: [torch.Tensor], labels: [torch.Tensor], ins_ids: [torch.Tensor], smpl_ids: torch.Tensor,
               conf_scores: [torch.Tensor], iteration: int) -> None:
        for i in range(len(feats)):
            self.feats.append(feats[i])                 # (N, C)
            self.labels.append(labels[i].view(-1))      # (N,)
            self.ins_ids.append(ins_ids[i].view(-1))    # (N,)
            self.smpl_ids.append(smpl_ids[i].view(-1))  # (1,)     
            self.conf_scores.append(conf_scores[i].view(-1))  # (N,)
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
            conf_scores = torch.cat((self.conf_scores,),dim=0)
        except:
            features = torch.cat((self.feats), dim=0)
            ins_ids = torch.cat(self.ins_ids).int().cpu().numpy()
            labels = torch.cat((self.labels), dim=0).int()
            iterations = torch.cat(self.iterations).int().cpu().numpy()
            ins_ids = torch.cat((self.ins_ids), dim=0).int().cpu().numpy()
            iterations = torch.cat((self.iterations), dim=0).int().cpu().numpy()            
            conf_scores = torch.cat((self.conf_scores),dim=0)
            # pl_labels = torch.cat((self.pl_labels),dim=0).int()

        assert len(features) == len(labels) == len(ins_ids) == len(iterations) == len(conf_scores), \
            "length of features, labels, ins_ids, conf_scores and iterations should be the same"
        sorted_ins_ids, arg_sorted_ins_ids = np.sort(ins_ids), np.argsort(ins_ids)
        unique_ins_ids, split_indices = np.unique(sorted_ins_ids, return_index=True)
        self.proto_conf_scores = conf_scores[arg_sorted_ins_ids[split_indices]] # Update proto_conf_scores at every compute() call -> with fresh batch's conf_scores

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
    

    def topk_padding(self, pseudo_positive_labels, pseudo_positives, topk_list, pseudo_conf_scores, class_labels=(1, 2, 3)):
        for k, class_label in enumerate(class_labels):
            if torch.nonzero(pseudo_positive_labels == class_label).shape[0] < topk_list[k]:
                num_to_append = topk_list[k] - torch.nonzero(pseudo_positive_labels == class_label).shape[0]

                labels = torch.tensor([class_label] * num_to_append).to(pseudo_positives.device)
                pseudo_positive_labels = torch.cat((pseudo_positive_labels, labels), dim=0)
                
                synthetic_samples = self.generate_synthetic_samples(
                    self.classwise_prototypes[k].unsqueeze(0).expand(num_to_append, -1), noise_level=0.1
                )
                pseudo_positives = torch.cat((pseudo_positives, synthetic_samples), dim=0)
                if pseudo_conf_scores is not None:
                    pseudo_conf_scores = torch.cat((pseudo_conf_scores, (0.7 * torch.ones(num_to_append).to(pseudo_positives.device))), dim=0)
                    return pseudo_positive_labels, pseudo_positives, pseudo_conf_scores
                else:
                    return pseudo_positive_labels, pseudo_positives, None


    def sample_topk(self, pseudo_positive_labels, pseudo_positives, topk_list, pseudo_conf_scores):
        if pseudo_conf_scores is not None:
            car_idx = torch.where(pseudo_positive_labels==1)[0]

            car_conf_scores = torch.index_select(pseudo_conf_scores, dim=0, index=car_idx)
            car_positives = torch.index_select(pseudo_positives, dim=0, index=car_idx)
            car_labels= torch.index_select(pseudo_positive_labels, dim=0, index=car_idx)

            topk_conf_cars, topk_car_idx = torch.topk(car_conf_scores,topk_list[0])
            topk_car_fts = torch.index_select(car_positives, 0, topk_car_idx)
            topk_car_labels = torch.index_select(car_labels, 0, topk_car_idx)

            ped_idx = torch.where(pseudo_positive_labels==2)[0]
            ped_conf_scores = torch.index_select(pseudo_conf_scores, dim=0, index=ped_idx)
            ped_positives = torch.index_select(pseudo_positives, dim=0, index=ped_idx)
            ped_labels= torch.index_select(pseudo_positive_labels, dim=0, index=ped_idx)

            topk_conf_peds, topk_ped_idx = torch.topk(ped_conf_scores,topk_list[1])
            topk_ped_fts = torch.index_select(ped_positives, dim=0, index=topk_ped_idx)
            topk_ped_labels = torch.index_select(ped_labels, dim=0, index=topk_ped_idx)

            cyc_idx = torch.where(pseudo_positive_labels==3)[0]
            cyc_conf_scores = torch.index_select(pseudo_conf_scores, dim=0, index=cyc_idx)
            cyc_positives = torch.index_select(pseudo_positives, dim=0, index=cyc_idx)
            cyc_labels= torch.index_select(pseudo_positive_labels, dim=0, index=cyc_idx)

            topk_conf_cycs, topk_cyc_idx = torch.topk(cyc_conf_scores,topk_list[2])
            topk_cyc_fts = torch.index_select(cyc_positives, dim=0, index=topk_cyc_idx)
            topk_cyc_labels = torch.index_select(cyc_labels, dim=0, index=topk_cyc_idx)

            pseudo_topk_labels = torch.cat((topk_car_labels,topk_ped_labels,topk_cyc_labels),dim=0)
            pseudo_topk_fts = torch.cat((topk_car_fts, topk_ped_fts, topk_cyc_fts), dim=0)

        else:
            car_mk = torch.topk((torch.where(pseudo_positive_labels==1)[0]),topk_list[0])[0]
            topk_cars = pseudo_positives[car_mk]
            ped_mk = torch.topk((torch.where(pseudo_positive_labels==2)[0]),topk_list[1])[0]
            topk_peds = pseudo_positives[ped_mk]
            cyc_mk = torch.topk((torch.where(pseudo_positive_labels==3)[0]),topk_list[2])[0]
            topk_cycs = pseudo_positives[cyc_mk]
            pseudo_topk_labels = torch.cat((pseudo_positive_labels[car_mk],pseudo_positive_labels[ped_mk],pseudo_positive_labels[cyc_mk]),dim=0)
            pseudo_topk_fts = torch.cat((topk_cars, topk_peds, topk_cycs),dim=0)
        
        return pseudo_topk_labels,pseudo_topk_fts

    # def get_uniform_samples(self, prototypes, labels, unique_labels,num_per_class):
    #     indices = []
    #     for label in unique_labels[1:]:  # Assuming labels are 1, 2, or 3
    #         label_indices = torch.topk((torch.where(labels == label)[0]),num_per_class)[0]
    #         indices.append(label_indices)
    #     # Concatenate the indices for each class and select from prototypes
    #     indices = torch.cat((indices), dim=0)
    #     uniform_prototypes = prototypes.index_select(0, indices)
    #     uniform_labels = labels.index_select(0, indices)
    #     return uniform_prototypes, uniform_labels

    # def informative_car_features(self, car_prototypes, car_labels):
    #     kmeans = KMeans(n_clusters=9, random_state=0).fit(car_prototypes.cpu().numpy())
    #     cluster_labels = kmeans.labels_
    #     centroids = kmeans.cluster_centers_ # Using centroids as the informative car features
    #     car_centroids = torch.from_numpy(centroids).to(car_prototypes.device)
    #     return car_centroids, car_labels[:9]


    def get_computed_protos(self):
        return  self.classwise_prototypes, self.prototypes, self.proto_labels, self.num_updates
    
    def generate_synthetic_samples(self, batch_tensors, noise_level=0.0001):
        noise = torch.randn_like(batch_tensors) * noise_level
        new_samples = batch_tensors + noise
        new_samples = torch.abs(new_samples)
        return new_samples

    def get_lpcont_loss_ori(self, pseudo_positives, pseudo_positive_labels, topk_list, epoch, CLIP_CE=False):
        """
        param pseudo_positives: Pseudo positive student features(Mk, Channel=256)
        param topk_labels: Labels for pseudo positive student features
        return:
        """
        N = len(self.prototypes)  #161
        contrastive_loss = torch.tensor(0.0).to(pseudo_positives.device) #contrastive_loss2 = torch.tensor(0.0).to(pseudo_positives.device)
        labeled_bank_info = torch.cat([self.prototypes, self.proto_labels.unsqueeze(-1)], dim=-1)
        pseudo_batch_info = torch.cat([pseudo_positives, pseudo_positive_labels.unsqueeze(-1)], dim=-1)

        gathered_lbl_tensor = self.gather_tensors(labeled_bank_info)
        gathered_sa_labels = gathered_lbl_tensor[:,-1].long()
        non_zero_mask = gathered_sa_labels != 0
        gathered_prototypes = gathered_lbl_tensor[:,:-1][non_zero_mask]
        gathered_labels = gathered_sa_labels[non_zero_mask]
        gathered_conf_scores = gathered_lbl_tensor[:,-2][non_zero_mask]

        gathered_pseudo_tensor = self.gather_tensors(pseudo_batch_info)
        gathered_wa_labels = gathered_pseudo_tensor[:,-1].long()
        non_zero_mask2 = gathered_wa_labels != 0
        gathered_pseudo_positives = gathered_pseudo_tensor[:,:-1][non_zero_mask2]
        gathered_pseudo_labels = gathered_wa_labels[non_zero_mask2]

        sorted_labels, sorted_args = torch.sort(gathered_labels) #161
        sorted_prototypes = gathered_prototypes[sorted_args] # sort prototypes to arrange classwise #161
        # sorted_conf_scores = gathered_conf_scores[sorted_args]

        # '''Append classwise prototypes to pseudo_labels'''
        # gathered_pseudo_positives = torch.cat((gathered_pseudo_positives,self.classwise_prototypes),dim=0)
        # gathered_pseudo_labels = torch.cat((gathered_pseudo_labels,self.proto_labels.unique()), dim=0)
        
        sorted_pls, pl_args = torch.sort(gathered_pseudo_labels)
        sorted_pseudo_positives = gathered_pseudo_positives[pl_args]
        # self.temperature= self.initial_temperature - (self.initial_temperature - self.final_temperature) * (epoch / self.epochs)
        self.temperature = 1.0
        # self.weight = self.initial_weight + (self.final_weight - self.initial_weight) * (epoch / self.epochs)          
        pseudo_conf_scores = None # Ori_GTs, do not use pseudo_conf_score for filtering - pass all
        #DISABLE topk_padding and topk_sampling
        # padded_pls, padded_pseudo_positives, padded_pseudo_conf_scores = self.topk_padding(sorted_pls, sorted_pseudo_positives, topk_list, pseudo_conf_scores)
        # sorted_pls, sorted_pseudo_positives = self.sample_topk(padded_pls, padded_pseudo_positives, topk_list, padded_pseudo_conf_scores)
        # non_car_pl_mask = torch.where(sorted_pls!=1)[0]
        # non_car_lbl_mask = torch.where(sorted_labels!=1)[0]
        # minority_labels = sorted_labels[non_car_lbl_mask]
        # minority_pls = sorted_pls[non_car_pl_mask]
        # minority_prototypes = sorted_prototypes[non_car_lbl_mask]
        # minority_pseudo_positives = sorted_pseudo_positives[non_car_pl_mask]
        # label_mask = minority_labels.unsqueeze(1)== minority_pls.unsqueeze(0)  # Shape: 27,15
        # positive_mask = label_mask #(27,15)
        # negative_mask = ~label_mask #(27,15)
        # uniform_sorted_prototypes = F.normalize(minority_prototypes,dim=-1) #27,256
        # pseudo_topk_features = F.normalize(minority_pseudo_positives,dim=-1) #15,256
        # sim_matrix = uniform_sorted_prototypes @ pseudo_topk_features.t() # (27,256) @ (256,15) -> (27,15)
        # proto_conf_scores = self.proto_conf_scores[non_car_lbl_mask].detach().cpu().numpy() 
        label_mask = sorted_labels.unsqueeze(1)== sorted_pls.unsqueeze(0)  # Shape: 27,15
        positive_mask = label_mask #(27,15)
        negative_mask = ~label_mask #(27,15)
        uniform_sorted_prototypes = F.normalize(sorted_prototypes,dim=-1) #27,256
        pseudo_topk_features = F.normalize(sorted_pseudo_positives,dim=-1) #15,256
        sim_matrix = uniform_sorted_prototypes @ pseudo_topk_features.t() # (27,256) @ (256,15) -> (27,15)
        proto_conf_scores = self.proto_conf_scores.detach().cpu().numpy() 
        exp_sim_pos_matrix = torch.exp(sim_matrix * positive_mask / self.temperature)
        positive_sum = torch.log(exp_sim_pos_matrix).sum() # Sum of log(exp(positives))
        positive_sum_mean = positive_sum.mean()
        negative_sum = (torch.exp(sim_matrix) * negative_mask).sum(dim=0, keepdims=True) #/((unique_labels.size(0)-1) * counts.min())
        pairwise_negative_sum =  torch.log(negative_sum).sum()
        negative_sum_mean = pairwise_negative_sum.mean()
        unscaled_contrastive_loss = positive_sum - pairwise_negative_sum
        # unscaled_contrastive_loss = positive_sum 
        contrastive_loss = contrastive_loss +  -1 * (unscaled_contrastive_loss / (sorted_prototypes.size(0) * pseudo_topk_features.size(0) * 3))
        return contrastive_loss,  (sim_matrix, sorted_labels, sorted_pls, proto_conf_scores), positive_sum_mean, negative_sum_mean

    def get_lpcont_loss_pls(self, pseudo_positives, pseudo_positive_labels, topk_list, pseudo_conf_scores = None, CLIP_CE=False):
        """
        param pseudo_positives: Pseudo positive student features(Mk, Channel=256)
        param topk_labels: Labels for pseudo positive student features
        return:
        """
        N = len(self.prototypes)
        contrastive_loss = torch.tensor(0.0).to(pseudo_positives.device) #contrastive_loss2 = torch.tensor(0.0).to(pseudo_positives.device)

        labeled_bank_info = torch.cat([self.prototypes, self.proto_conf_scores.unsqueeze(-1), self.proto_labels.unsqueeze(-1)], dim=-1)
        pseudo_batch_info = torch.cat([pseudo_positives, pseudo_conf_scores.unsqueeze(-1), pseudo_positive_labels.unsqueeze(-1)], dim=-1)

        gathered_lbl_tensor = self.gather_tensor_pls(labeled_bank_info)
        gathered_sa_labels = gathered_lbl_tensor[:,-1].long()
        non_zero_mask = gathered_sa_labels != 0
        gathered_prototypes = gathered_lbl_tensor[:,:-2][non_zero_mask]
        gathered_labels = gathered_sa_labels[non_zero_mask]
        gathered_conf_scores = gathered_lbl_tensor[:,-2][non_zero_mask]
        
        gathered_pseudo_tensor = self.gather_tensor_pls(pseudo_batch_info)
        gathered_wa_labels = gathered_pseudo_tensor[:,-1].long()
        non_zero_mask2 = gathered_wa_labels != 0
        gathered_pseudo_positives = gathered_pseudo_tensor[:,:-2][non_zero_mask2]
        gathered_pseudo_labels = gathered_wa_labels[non_zero_mask2]
        gathered_pseudo_conf_scores = gathered_pseudo_tensor[:,-2][non_zero_mask2]


        sorted_labels, sorted_args = torch.sort(gathered_labels) #161
        sorted_prototypes = gathered_prototypes[sorted_args] # sort prototypes to arrange classwise #161
        sorted_conf_scores = gathered_conf_scores[sorted_args]
        # unique_labels, counts = torch.unique(sorted_labels, return_counts=True)
        
        sorted_pls, pl_args = torch.sort(gathered_pseudo_labels)
        sorted_pseudo_positives = gathered_pseudo_positives[pl_args]
        sorted_pseudo_conf_scores = gathered_pseudo_conf_scores[pl_args]
        '''NO padding for Pseudo Labels'''
        # sorted_pls, sorted_pseudo_positives, sorted_pseudo_conf_scores = self.topk_padding(sorted_pls, sorted_pseudo_positives, topk_list, sorted_pseudo_conf_scores)
        # sampled_pls, sampled_pseudo_positives = self.sample_topk(sorted_pls, sorted_pseudo_positives, topk_list, sorted_pseudo_conf_scores)
        '''Append classwise prototypes to pseudo_labels'''
        gathered_pseudo_positives = torch.cat((gathered_pseudo_positives,self.classwise_prototypes),dim=0)
        gathered_pseudo_labels = torch.cat((gathered_pseudo_labels,self.proto_labels.unique()), dim=0)
        
        label_mask = sorted_labels.unsqueeze(1)== sorted_pls.unsqueeze(0)
        positive_mask = label_mask
        negative_mask = ~label_mask           
        ## Product of conf_scores and pseudo_conf_scores as weights for negative_mask
        # confidence_weights = sorted_conf_scores.unsqueeze(1) * sorted_pseudo_conf_scores.unsqueeze(0)

        norm_sorted_prototypes = F.normalize(sorted_prototypes, dim=-1)
        norm_pseudo_topk_features = F.normalize(sorted_pseudo_positives, dim=-1)
        sim_matrix = norm_sorted_prototypes @ norm_pseudo_topk_features.t()
        self.temperature = 1.0
        exp_sim_pos_matrix = torch.exp(sim_matrix * positive_mask /self.temperature)
        # sim_pos_matrix_row = exp_sim_pos_matrix.clone()
        positive_sum = torch.log(exp_sim_pos_matrix).sum() # Sum of log(exp(positives))
        # number_negative_pairs = negative_mask.sum(dim=0, keepdims=True) # Number of negative pairs for each positive
        negative_sum = (torch.exp(sim_matrix) * negative_mask).sum(dim=0, keepdims=True) #/((unique_labels.size(0)-1) * counts.min())
        pairwise_negatives_sum =  torch.log(negative_sum).sum()
        #unscaled_contrastive_loss = positive_sum 
        unscaled_contrastive_loss = ((positive_sum - pairwise_negatives_sum)) 
        contrastive_loss = contrastive_loss +  -1 * (unscaled_contrastive_loss / (sorted_prototypes.size(0) * norm_pseudo_topk_features.size(0) * 3))            
        
        return contrastive_loss, (sim_matrix, sorted_labels, sorted_pls, sorted_conf_scores), positive_sum, pairwise_negatives_sum

    def gather_tensors(self, tensor):
            """
            Returns the gathered tensor to all GPUs in DDP else returns the tensor as such
            dist.gather_all needs the gathered tensors to be of same size.
            We get the sizes of the tensors first, zero pad them to match the size
            Then gather and filter the padding
            Args:
                tensor: tensor to be gathered
                
            """

            assert tensor.size(-1) == 257 , "features should be of size common_instances,(ft_size+ lbl_size)"
            if not dist.is_initialized():
                return tensor
                # Determine sizes first
            WORLD_SIZE = dist.get_world_size()
            local_size = torch.tensor(tensor.size(), device=tensor.device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(WORLD_SIZE)]

            dist.barrier() 
            dist.all_gather(all_sizes,local_size)
            dist.barrier()

            print(f'all_sizes {all_sizes}')
            # make zero-padded version https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
            max_length = max([size[0] for size in all_sizes])

            diff = max_length - local_size[0].item()
            if diff:
                pad_size =[diff.item()] #pad with zeros 
                if local_size.ndim >= 1:
                    pad_size.extend(dimension.item() for dimension in local_size[1:])
                padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat((tensor,padding),)

            all_tensors_padded = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]

            dist.barrier()
            dist.all_gather(all_tensors_padded,tensor)
            dist.barrier()

            gathered_tensor = torch.cat(all_tensors_padded)
            non_zero_mask = torch.any(gathered_tensor!=0,dim=-1).squeeze()
            gathered_tensor = gathered_tensor[non_zero_mask]
            return gathered_tensor

    def gather_tensor_pls(self, tensor):
            """
            Returns the gathered tensor to all GPUs in DDP else returns the tensor as such
            dist.gather_all needs the gathered tensors to be of same size.
            We get the sizes of the tensors first, zero pad them to match the size
            Then gather and filter the padding
            Args:
                tensor: tensor to be gathered
                
            """

            assert tensor.size(-1) == 258 , "features should be of size common_instances,(ft_size+ conf_score + lbl_size)"
            if not dist.is_initialized():
                return tensor
                # Determine sizes first
            WORLD_SIZE = dist.get_world_size()
            local_size = torch.tensor(tensor.size(), device=tensor.device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(WORLD_SIZE)]

            dist.barrier() 
            dist.all_gather(all_sizes,local_size)
            dist.barrier()

            print(f'all_sizes {all_sizes}')
            # make zero-padded version https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
            max_length = max([size[0] for size in all_sizes])

            diff = max_length - local_size[0].item()
            if diff:
                pad_size =[diff.item()] #pad with zeros 
                if local_size.ndim >= 1:
                    pad_size.extend(dimension.item() for dimension in local_size[1:])
                padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat((tensor,padding),)

            all_tensors_padded = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]

            dist.barrier()
            dist.all_gather(all_tensors_padded,tensor)
            dist.barrier()

            gathered_tensor = torch.cat(all_tensors_padded)
            non_zero_mask = torch.any(gathered_tensor!=0,dim=-1).squeeze()
            gathered_tensor = gathered_tensor[non_zero_mask]
            return gathered_tensor


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