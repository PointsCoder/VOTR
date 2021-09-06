import torch
import torch.nn as nn

from ...ops.votr_ops import votr_utils

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

class SparseTensor(object):
    def __init__(self, features, indices, spatial_shape, voxel_size, point_cloud_range, batch_size, hash_size, map_table = None, gather_dict = None):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape # [x, y, z]
        self.batch_size = batch_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.hash_size = hash_size
        self.gather_dict = gather_dict
        self.map_table = self.build_map_table() if not map_table else map_table

    @torch.no_grad()
    def build_map_table(self):
        bs_cnt = torch.zeros(self.batch_size).int()
        for i in range(self.batch_size):
            bs_cnt[i] = (self.indices[:, 0] == i).sum().item()
        bs_cnt = bs_cnt.to(self.indices.device)
        map_table = votr_utils.build_hash_table(
            self.batch_size,
            self.hash_size,
            self.spatial_shape,
            self.indices,
            bs_cnt,
        )
        return map_table

    def dense(self, channels_first=True):
        reverse_spatial_shape = self.spatial_shape[::-1] # (ZYX)
        output_shape = [self.batch_size] + list(
            reverse_spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(
            self.indices.to(self.features.device).long(), self.features,
            output_shape)
        if not channels_first:
            return res
        ndim = len(reverse_spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

class Attention3d(nn.Module):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes):
        super(Attention3d, self).__init__()
        self.attention_modes = attention_modes

        self.mhead_attention = nn.MultiheadAttention(
                embed_dim= input_channels,
                num_heads= num_heads,
                dropout= dropout,
                )
        self.drop_out = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_channels, ff_channels)
        self.linear2 = nn.Linear(ff_channels, input_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.output_layer = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )

    @torch.no_grad()
    def with_bs_cnt(self, indices, batch_size):
        bs_cnt = torch.zeros(batch_size).int()
        for i in range(batch_size):
            bs_cnt[i] = (indices[:, 0] == i).sum().item()
        bs_cnt = bs_cnt.to(indices.device)
        return bs_cnt

    @torch.no_grad()
    def with_coords(self, indices, point_cloud_range, voxel_size):
        voxel_size = torch.tensor(voxel_size).unsqueeze(0).to(indices.device)
        min_range = torch.tensor(point_cloud_range[0:3]).unsqueeze(0).to(indices.device)
        coords = (indices[:, [3, 2, 1]].float() + 0.5) * voxel_size + min_range
        return coords

    def forward(self, sp_tensor):
        raise NotImplementedError

class SparseAttention3d(Attention3d):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes, strides, num_ds_voxels,
                 use_relative_coords = False, use_pooled_feature = False, use_no_query_coords = False):
        super(SparseAttention3d, self).__init__(input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes)

        self.use_relative_coords = use_relative_coords
        self.use_pooled_features = use_pooled_feature
        self.use_no_query_coords = use_no_query_coords

        self.strides = strides
        self.num_ds_voxels = num_ds_voxels

        self.norm = nn.BatchNorm1d(input_channels)
        if not self.use_no_query_coords:
            self.q_pos_proj = nn.Sequential(
                nn.Linear(3, input_channels),
                nn.ReLU(),
            )
        self.k_pos_proj = nn.Sequential(
            nn.Conv1d(3, input_channels, 1),
            nn.ReLU(),
        )

    @torch.no_grad()
    def create_gather_dict(self, attention_modes, map_table, voxel_indices, spatial_shape):
        _gather_dict = {}
        for attention_mode in attention_modes:
            if attention_mode.NAME == 'LocalAttention':
                attend_size = attention_mode.SIZE
                attend_range = attention_mode.RANGE
                _gather_indices = votr_utils.sparse_local_attention_hash_indices(spatial_shape, attend_size, attend_range, self.strides, map_table, voxel_indices)
            elif attention_mode.NAME == 'StridedAttention':
                attend_size = attention_mode.SIZE
                range_spec = attention_mode.RANGE_SPEC
                _gather_indices = votr_utils.sparse_strided_attention_hash_indices(spatial_shape, attend_size, range_spec, self.strides, map_table, voxel_indices)
            else:
                raise NotImplementedError

            _gather_mask = (_gather_indices < 0)
            #_gather_indices[_gather_indices < 0] = 0
            _gather_dict[attention_mode.NAME] = [_gather_indices, _gather_mask]

        return _gather_dict

    @torch.no_grad()
    def downsample(self, sp_tensor):
        x_shape = sp_tensor.spatial_shape[0] // self.strides[0]
        y_shape = sp_tensor.spatial_shape[1] // self.strides[1]
        z_shape = sp_tensor.spatial_shape[2] // self.strides[2]
        new_spatial_shape = [x_shape, y_shape, z_shape]
        new_indices, new_map_table = votr_utils.hash_table_down_sample(self.strides, self.num_ds_voxels, sp_tensor.batch_size, sp_tensor.hash_size, new_spatial_shape, sp_tensor.indices)
        return new_spatial_shape, new_indices, new_map_table

    def forward(self, sp_tensor):
        new_spatial_shape, new_indices, new_map_table = self.downsample(sp_tensor)
        vx, vy, vz = sp_tensor.voxel_size
        new_voxel_size = [vx * self.strides[0], vy * self.strides[1], vz * self.strides[2]]
        gather_dict = self.create_gather_dict(self.attention_modes, sp_tensor.map_table, new_indices, sp_tensor.spatial_shape)

        voxel_features = sp_tensor.features
        v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size)
        k_bs_cnt = self.with_bs_cnt(new_indices, sp_tensor.batch_size)

        a_key_indices, a_key_mask = [], []
        for attention_idx, attetion_mode in enumerate(self.attention_modes):
            key_indices, key_mask = gather_dict[attetion_mode.NAME]
            a_key_indices.append(key_indices)
            a_key_mask.append(key_mask)

        key_indices = torch.cat(a_key_indices, dim = 1)
        key_mask = torch.cat(a_key_mask, dim = 1)

        key_features = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices, k_bs_cnt)
        voxel_coords = self.with_coords(sp_tensor.indices, sp_tensor.point_cloud_range, sp_tensor.voxel_size)
        key_coords = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, key_indices, k_bs_cnt)

        query_coords = self.with_coords(new_indices, sp_tensor.point_cloud_range, new_voxel_size)

        if self.use_pooled_features:
            pooled_query_features = key_features.max(dim=-1)[0]
            pooled_query_features = pooled_query_features.unsqueeze(0)
            if self.use_no_query_coords:
                query_features = pooled_query_features
            else:
                query_features = self.q_pos_proj(query_coords).unsqueeze(0)
                query_features = query_features + pooled_query_features
        else:
            query_features = self.q_pos_proj(query_coords).unsqueeze(0)

        if self.use_relative_coords:
            key_coords = key_coords - query_coords.unsqueeze(-1) # (N, 3, size)

        key_pos_emb = self.k_pos_proj(key_coords)
        key_features = key_features + key_pos_emb
        key_features = key_features.permute(2, 0, 1).contiguous() # (size, N1+N2, C)

        attend_features, attend_weights = self.mhead_attention(
            query = query_features,
            key = key_features,
            value = key_features,
            key_padding_mask = key_mask,
        )

        attend_features = self.drop_out(attend_features)

        new_features = attend_features.squeeze(0)
        act_features = self.linear2(self.dropout1(self.activation(self.linear1(new_features))))
        new_features = new_features + self.dropout2(act_features)
        new_features = self.norm(new_features)
        new_features = self.output_layer(new_features)

        # update sp_tensor
        sp_tensor.features = new_features
        sp_tensor.indices = new_indices
        sp_tensor.spatial_shape = new_spatial_shape
        sp_tensor.voxel_size = new_voxel_size

        del sp_tensor.map_table
        sp_tensor.gather_dict = None
        sp_tensor.map_table = new_map_table
        return sp_tensor

class SubMAttention3d(Attention3d):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes,
                 use_pos_emb = True, use_relative_coords = False, use_no_query_coords = False):
        super(SubMAttention3d, self).__init__(input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes)

        self.use_relative_coords = use_relative_coords
        self.use_no_query_coords = use_no_query_coords
        self.use_pos_emb = use_pos_emb

        self.norm1 = nn.BatchNorm1d(input_channels)
        self.norm2 = nn.BatchNorm1d(input_channels)
        if self.use_pos_emb:
            if not self.use_no_query_coords:
                self.q_pos_proj = nn.Sequential(
                    nn.Linear(3, input_channels),
                    nn.ReLU(),
                )
            self.k_pos_proj = nn.Sequential(
                nn.Conv1d(3, input_channels, 1),
                nn.ReLU(),
            )

    @torch.no_grad()
    def create_gather_dict(self, attention_modes, map_table, voxel_indices, spatial_shape):
        _gather_dict = {}
        for attention_mode in attention_modes:
            if attention_mode.NAME == 'LocalAttention':
                attend_size = attention_mode.SIZE
                attend_range = attention_mode.RANGE
                _gather_indices = votr_utils.subm_local_attention_hash_indices(spatial_shape, attend_size, attend_range, map_table, voxel_indices)
            elif attention_mode.NAME == 'StridedAttention':
                attend_size = attention_mode.SIZE
                range_spec = attention_mode.RANGE_SPEC
                _gather_indices = votr_utils.subm_strided_attention_hash_indices(spatial_shape, attend_size, range_spec, map_table, voxel_indices)
            else:
                raise NotImplementedError

            _gather_mask = (_gather_indices < 0)
            #_gather_indices[_gather_indices < 0] = 0
            _gather_dict[attention_mode.NAME] = [_gather_indices, _gather_mask]

        return _gather_dict

    def forward(self, sp_tensor):
        if not sp_tensor.gather_dict:
            sp_tensor.gather_dict = self.create_gather_dict(self.attention_modes, sp_tensor.map_table, sp_tensor.indices, sp_tensor.spatial_shape)

        voxel_features = sp_tensor.features
        v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size)
        k_bs_cnt = v_bs_cnt.clone()

        a_key_indices, a_key_mask = [], []
        for attention_idx, attetion_mode in enumerate(self.attention_modes):
            key_indices, key_mask = sp_tensor.gather_dict[attetion_mode.NAME]
            a_key_indices.append(key_indices)
            a_key_mask.append(key_mask)

        key_indices = torch.cat(a_key_indices, dim = 1)
        key_mask = torch.cat(a_key_mask, dim = 1)

        query_features = voxel_features.unsqueeze(0) # (1, N1+N2, C)
        key_features = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices, k_bs_cnt)

        if self.use_pos_emb:
            voxel_coords = self.with_coords(sp_tensor.indices, sp_tensor.point_cloud_range, sp_tensor.voxel_size)
            key_coords = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, key_indices, k_bs_cnt)
            if self.use_relative_coords:
                key_coords = key_coords - voxel_coords.unsqueeze(-1)
            key_pos_emb = self.k_pos_proj(key_coords)
            key_features = key_features + key_pos_emb

            if self.use_no_query_coords:
                pass
            else:
                query_pos_emb = self.q_pos_proj(voxel_coords).unsqueeze(0)
                query_features = query_features + query_pos_emb

        key_features = key_features.permute(2, 0, 1).contiguous() # (size, N1+N2, C)

        attend_features, attend_weights = self.mhead_attention(
            query = query_features,
            key = key_features,
            value = key_features,
            key_padding_mask = key_mask,
        )

        attend_features = self.drop_out(attend_features)
        voxel_features = voxel_features + attend_features.squeeze(0)
        voxel_features = self.norm1(voxel_features)
        act_features = self.linear2(self.dropout1(self.activation(self.linear1(voxel_features))))
        voxel_features = voxel_features + self.dropout2(act_features)
        voxel_features = self.norm2(voxel_features)
        voxel_features = self.output_layer(voxel_features)
        sp_tensor.features = voxel_features
        return sp_tensor

class AttentionResBlock(nn.Module):
    def __init__(self, model_cfg, use_relative_coords = False, use_pooled_feature = False, use_no_query_coords = False):
        super(AttentionResBlock, self).__init__()
        sp_cfg = model_cfg.SP_CFGS
        self.sp_attention = SparseAttention3d(
            input_channels = sp_cfg.CHANNELS[0],
            output_channels = sp_cfg.CHANNELS[2],
            ff_channels = sp_cfg.CHANNELS[1],
            dropout = sp_cfg.DROPOUT,
            num_heads = sp_cfg.NUM_HEADS,
            attention_modes = sp_cfg.ATTENTION,
            strides = sp_cfg.STRIDE,
            num_ds_voxels = sp_cfg.NUM_DS_VOXELS,
            use_relative_coords = use_relative_coords,
            use_pooled_feature = use_pooled_feature,
            use_no_query_coords= use_no_query_coords,
        )
        subm_cfg = model_cfg.SUBM_CFGS
        self.subm_attention_modules = nn.ModuleList()
        for i in range(subm_cfg.NUM_BLOCKS):
            self.subm_attention_modules.append(SubMAttention3d(
                input_channels = subm_cfg.CHANNELS[0],
                output_channels = subm_cfg.CHANNELS[2],
                ff_channels = subm_cfg.CHANNELS[1],
                dropout = subm_cfg.DROPOUT,
                num_heads = subm_cfg.NUM_HEADS,
                attention_modes = subm_cfg.ATTENTION,
                use_pos_emb =  subm_cfg.USE_POS_EMB,
                use_relative_coords = use_relative_coords,
                use_no_query_coords= use_no_query_coords,
            ))

    def forward(self, sp_tensor):
        sp_tensor = self.sp_attention(sp_tensor)
        indentity_features = sp_tensor.features
        for subm_module in self.subm_attention_modules:
            sp_tensor = subm_module(sp_tensor)
        sp_tensor.features += indentity_features
        return sp_tensor

class VoxelTransformer(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range):
        super(VoxelTransformer, self).__init__()
        self.model_cfg = model_cfg

        self.use_relative_coords = self.model_cfg.get('USE_RELATIVE_COORDS', False)
        self.use_pooled_feature = self.model_cfg.get('USE_POOLED_FEATURE', False)
        self.use_no_query_coords = self.model_cfg.get('USE_NO_QUERY_COORDS', False)

        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.input_transform = nn.Sequential(
            nn.Linear(input_channels, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.backbone = nn.ModuleList()
        for param in self.model_cfg.PARAMS:
            self.backbone.append(AttentionResBlock(param, self.use_relative_coords, self.use_pooled_feature, self.use_no_query_coords))

        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        voxel_features = self.input_transform(voxel_features)

        sp_tensor = SparseTensor(
            features = voxel_features,
            indices = voxel_coords.int(),
            spatial_shape = self.grid_size,
            voxel_size = self.voxel_size,
            point_cloud_range = self.point_cloud_range,
            batch_size = batch_size,
            hash_size = self.model_cfg.HASH_SIZE,
            map_table = None,
            gather_dict = None,
        )
        for attention_block in self.backbone:
            sp_tensor = attention_block(sp_tensor)

        batch_dict.update({
            'encoded_spconv_tensor': sp_tensor,
            'encoded_spconv_tensor_stride': 8
        })
        return batch_dict

class SparseAttention3dv2(Attention3d):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes, strides, num_ds_voxels,
                 use_relative_coords = False, use_pooled_feature = False, use_no_query_coords = False, global_mode = None):
        super(SparseAttention3dv2, self).__init__(input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes)

        self.use_relative_coords = use_relative_coords
        self.use_pooled_features = use_pooled_feature
        self.use_no_query_coords = use_no_query_coords

        self.strides = strides
        self.num_ds_voxels = num_ds_voxels

        self.norm = nn.BatchNorm1d(input_channels)
        if not self.use_no_query_coords:
            self.q_pos_proj = nn.Sequential(
                nn.Linear(3, input_channels),
                nn.ReLU(),
            )
        self.k_pos_proj = nn.Sequential(
            nn.Conv1d(3, input_channels, 1),
            nn.ReLU(),
        )

        self.global_mode = global_mode
        if self.global_mode:
            self.global_feat2coords = nn.Sequential(
                nn.Linear(input_channels, input_channels),
                nn.BatchNorm1d(input_channels),
                nn.ReLU(),
                nn.Linear(input_channels, input_channels),
                nn.BatchNorm1d(input_channels),
                nn.ReLU(),
                nn.Linear(input_channels, 3)
            )

    def global_attention(self, query_features, query_indices, batch_size):
        """
        Args:
            query_features: (N, C)
            query_indices: (N, 4)
        Returns:
            global_features: (N, K, C) K is global size
            global_coords: (N, K, 3)
        """
        assert self.global_mode is not None
        K = self.global_mode.SIZE

        global_features, global_coords = [], []
        for batch_idx in range(batch_size):
            query_features_single = query_features[query_indices[:, 0]==batch_idx, :]
            num_voxels_single = query_features_single.shape[0]
            if self.global_mode.TYPE == 'TopK':
                global_features_single = torch.topk(query_features_single, K, dim = 0)[0] # (K, C)
                global_coords_single = self.global_feat2coords(global_features_single) # (K, 3)
            elif self.global_mode.TYPE == 'SVD':
                with torch.no_grad():
                    pca_features = query_features_single.transpose(0, 1).contiguous() # (C, N)
                    _, _, v = torch.pca_lowrank(pca_features, q=K) # (N, K)
                    global_features_single = torch.matmul(pca_features, v).transpose(0, 1).contiguous() # (K, C)
                global_coords_single = self.global_feat2coords(global_features_single)
            else:
                raise NotImplementedError
            global_features_single = global_features_single.transpose(0, 1).contiguous()  # (C, K)
            global_coords_single = global_coords_single.transpose(0, 1).contiguous()  # (3, K)
            global_features_single = global_features_single.unsqueeze(0).repeat(num_voxels_single, 1, 1) # (N, C, K)
            global_coords_single = global_coords_single.unsqueeze(0).repeat(num_voxels_single, 1, 1) # (N, 3, K)
            global_features.append(global_features_single)
            global_coords.append(global_coords_single)

        global_features = torch.cat(global_features, dim = 0)
        global_coords = torch.cat(global_coords, dim = 0)
        return global_features, global_coords

    @torch.no_grad()
    def create_gather_dict(self, attention_modes, map_table, voxel_indices, spatial_shape):
        _gather_dict = {}
        for attention_mode in attention_modes:
            if attention_mode.NAME == 'LocalAttention':
                attend_size = attention_mode.SIZE
                attend_range = attention_mode.RANGE
                _gather_indices = votr_utils.sparse_local_attention_hash_indices(spatial_shape, attend_size, attend_range, self.strides, map_table, voxel_indices)
            elif attention_mode.NAME == 'StridedAttention':
                attend_size = attention_mode.SIZE
                range_spec = attention_mode.RANGE_SPEC
                _gather_indices = votr_utils.sparse_strided_attention_hash_indices(spatial_shape, attend_size, range_spec, self.strides, map_table, voxel_indices)
            else:
                raise NotImplementedError

            _gather_mask = (_gather_indices < 0)
            #_gather_indices[_gather_indices < 0] = 0
            _gather_dict[attention_mode.NAME] = [_gather_indices, _gather_mask]

        return _gather_dict

    @torch.no_grad()
    def downsample(self, sp_tensor):
        x_shape = sp_tensor.spatial_shape[0] // self.strides[0]
        y_shape = sp_tensor.spatial_shape[1] // self.strides[1]
        z_shape = sp_tensor.spatial_shape[2] // self.strides[2]
        new_spatial_shape = [x_shape, y_shape, z_shape]
        new_indices, new_map_table = votr_utils.hash_table_down_sample(self.strides, self.num_ds_voxels, sp_tensor.batch_size, sp_tensor.hash_size, new_spatial_shape, sp_tensor.indices)
        return new_spatial_shape, new_indices, new_map_table

    def forward(self, sp_tensor):
        new_spatial_shape, new_indices, new_map_table = self.downsample(sp_tensor)
        vx, vy, vz = sp_tensor.voxel_size
        new_voxel_size = [vx * self.strides[0], vy * self.strides[1], vz * self.strides[2]]
        gather_dict = self.create_gather_dict(self.attention_modes, sp_tensor.map_table, new_indices, sp_tensor.spatial_shape)

        voxel_features = sp_tensor.features
        v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size)
        k_bs_cnt = self.with_bs_cnt(new_indices, sp_tensor.batch_size)

        a_key_indices, a_key_mask = [], []
        for attention_idx, attetion_mode in enumerate(self.attention_modes):
            key_indices, key_mask = gather_dict[attetion_mode.NAME]
            a_key_indices.append(key_indices)
            a_key_mask.append(key_mask)

        key_indices = torch.cat(a_key_indices, dim = 1)
        key_mask = torch.cat(a_key_mask, dim = 1)

        key_features = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices, k_bs_cnt)
        voxel_coords = self.with_coords(sp_tensor.indices, sp_tensor.point_cloud_range, sp_tensor.voxel_size)
        key_coords = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, key_indices, k_bs_cnt)

        query_coords = self.with_coords(new_indices, sp_tensor.point_cloud_range, new_voxel_size)

        if self.use_pooled_features:
            pooled_query_features = key_features.max(dim=-1)[0]
            pooled_query_features = pooled_query_features.unsqueeze(0)
            if self.use_no_query_coords:
                query_features = pooled_query_features
            else:
                query_features = self.q_pos_proj(query_coords).unsqueeze(0)
                query_features = query_features + pooled_query_features
        else:
            query_features = self.q_pos_proj(query_coords).unsqueeze(0)

        if self.global_mode:
            global_key_features, global_key_coords = self.global_attention(query_features=query_features.squeeze(0),
                                                                           query_indices=new_indices,
                                                                           batch_size=sp_tensor.batch_size)
            key_features = torch.cat([key_features, global_key_features], dim = 2) # (N, C, size+K)
            key_coords = torch.cat([key_coords, global_key_coords], dim = 2) # (N, 3, size+K)
            global_key_mask = torch.zeros((key_mask.shape[0], self.global_mode.SIZE), dtype=key_mask.dtype).to(key_mask.device)
            key_mask = torch.cat([key_mask, global_key_mask], dim = 1)

        if self.use_relative_coords:
            key_coords = key_coords - query_coords.unsqueeze(-1) # (N, 3, size)

        key_pos_emb = self.k_pos_proj(key_coords)
        key_features = key_features + key_pos_emb
        key_features = key_features.permute(2, 0, 1).contiguous() # (size, N1+N2, C)

        attend_features, attend_weights = self.mhead_attention(
            query = query_features,
            key = key_features,
            value = key_features,
            key_padding_mask = key_mask,
        )

        attend_features = self.drop_out(attend_features)

        new_features = attend_features.squeeze(0)
        act_features = self.linear2(self.dropout1(self.activation(self.linear1(new_features))))
        new_features = new_features + self.dropout2(act_features)
        new_features = self.norm(new_features)
        new_features = self.output_layer(new_features)

        # update sp_tensor
        sp_tensor.features = new_features
        sp_tensor.indices = new_indices
        sp_tensor.spatial_shape = new_spatial_shape
        sp_tensor.voxel_size = new_voxel_size

        del sp_tensor.map_table
        sp_tensor.gather_dict = None
        sp_tensor.map_table = new_map_table
        return sp_tensor

class SubMAttention3dv2(Attention3d):
    def __init__(self, input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes,
                 use_pos_emb = True, use_relative_coords = False, use_no_query_coords = False, global_mode = None):
        super(SubMAttention3dv2, self).__init__(input_channels, output_channels, ff_channels, dropout, num_heads, attention_modes)

        self.use_relative_coords = use_relative_coords
        self.use_no_query_coords = use_no_query_coords
        self.use_pos_emb = use_pos_emb

        self.norm1 = nn.BatchNorm1d(input_channels)
        self.norm2 = nn.BatchNorm1d(input_channels)
        if self.use_pos_emb:
            if not self.use_no_query_coords:
                self.q_pos_proj = nn.Sequential(
                    nn.Linear(3, input_channels),
                    nn.ReLU(),
                )
            self.k_pos_proj = nn.Sequential(
                nn.Conv1d(3, input_channels, 1),
                nn.ReLU(),
            )

        self.global_mode = global_mode
        if self.global_mode:
            self.global_feat2coords = nn.Sequential(
                nn.Linear(input_channels, input_channels),
                nn.BatchNorm1d(input_channels),
                nn.ReLU(),
                nn.Linear(input_channels, input_channels),
                nn.BatchNorm1d(input_channels),
                nn.ReLU(),
                nn.Linear(input_channels, 3)
            )

    def global_attention(self, query_features, query_indices, batch_size):
        """
        Args:
            query_features: (N, C)
            query_indices: (N, 4)
        Returns:
            global_features: (N, K, C) K is global size
            global_coords: (N, K, 3)
        """
        assert self.global_mode is not None
        K = self.global_mode.SIZE

        global_features, global_coords = [], []
        for batch_idx in range(batch_size):
            query_features_single = query_features[query_indices[:, 0]==batch_idx, :]
            num_voxels_single = query_features_single.shape[0]
            if self.global_mode.TYPE == 'TopK':
                global_features_single = torch.topk(query_features_single, K, dim = 0)[0] # (K, C)
                global_coords_single = self.global_feat2coords(global_features_single) # (K, 3)
            elif self.global_mode.TYPE == 'SVD':
                with torch.no_grad():
                    pca_features = query_features_single.transpose(0, 1).contiguous() # (C, N)
                    _, _, v = torch.pca_lowrank(pca_features, q=K) # (N, K)
                    global_features_single = torch.matmul(pca_features, v).transpose(0, 1).contiguous() # (K, C)
                global_coords_single = self.global_feat2coords(global_features_single)
            else:
                raise NotImplementedError
            global_features_single = global_features_single.transpose(0, 1).contiguous()  # (C, K)
            global_coords_single = global_coords_single.transpose(0, 1).contiguous()  # (3, K)
            global_features_single = global_features_single.unsqueeze(0).repeat(num_voxels_single, 1, 1) # (N, C, K)
            global_coords_single = global_coords_single.unsqueeze(0).repeat(num_voxels_single, 1, 1) # (N, 3, K)
            global_features.append(global_features_single)
            global_coords.append(global_coords_single)

        global_features = torch.cat(global_features, dim = 0)
        global_coords = torch.cat(global_coords, dim = 0)
        return global_features, global_coords

    @torch.no_grad()
    def create_gather_dict(self, attention_modes, map_table, voxel_indices, spatial_shape):
        _gather_dict = {}
        for attention_mode in attention_modes:
            if attention_mode.NAME == 'LocalAttention':
                attend_size = attention_mode.SIZE
                attend_range = attention_mode.RANGE
                _gather_indices = votr_utils.subm_local_attention_hash_indices(spatial_shape, attend_size, attend_range, map_table, voxel_indices)
            elif attention_mode.NAME == 'StridedAttention':
                attend_size = attention_mode.SIZE
                range_spec = attention_mode.RANGE_SPEC
                _gather_indices = votr_utils.subm_strided_attention_hash_indices(spatial_shape, attend_size, range_spec, map_table, voxel_indices)
            else:
                raise NotImplementedError

            _gather_mask = (_gather_indices < 0)
            #_gather_indices[_gather_indices < 0] = 0
            _gather_dict[attention_mode.NAME] = [_gather_indices, _gather_mask]

        return _gather_dict

    def forward(self, sp_tensor):
        if not sp_tensor.gather_dict:
            sp_tensor.gather_dict = self.create_gather_dict(self.attention_modes, sp_tensor.map_table, sp_tensor.indices, sp_tensor.spatial_shape)

        voxel_features = sp_tensor.features
        v_bs_cnt = self.with_bs_cnt(sp_tensor.indices, sp_tensor.batch_size)
        k_bs_cnt = v_bs_cnt.clone()

        a_key_indices, a_key_mask = [], []
        for attention_idx, attetion_mode in enumerate(self.attention_modes):
            key_indices, key_mask = sp_tensor.gather_dict[attetion_mode.NAME]
            a_key_indices.append(key_indices)
            a_key_mask.append(key_mask)

        key_indices = torch.cat(a_key_indices, dim = 1)
        key_mask = torch.cat(a_key_mask, dim = 1)

        query_features = voxel_features.unsqueeze(0) # (1, N1+N2, C)
        key_features = votr_utils.grouping_operation(voxel_features, v_bs_cnt, key_indices, k_bs_cnt)

        if self.global_mode:
            global_key_features, global_key_coords = self.global_attention(query_features=query_features.squeeze(0),
                                                                           query_indices=sp_tensor.indices,
                                                                           batch_size=sp_tensor.batch_size)
            key_features = torch.cat([key_features, global_key_features], dim = 2) # (N, C, size+K)
            global_key_mask = torch.zeros((key_mask.shape[0], self.global_mode.SIZE), dtype=key_mask.dtype).to(key_mask.device)
            key_mask = torch.cat([key_mask, global_key_mask], dim = 1)

        if self.use_pos_emb:
            voxel_coords = self.with_coords(sp_tensor.indices, sp_tensor.point_cloud_range, sp_tensor.voxel_size)
            key_coords = votr_utils.grouping_operation(voxel_coords, v_bs_cnt, key_indices, k_bs_cnt)

            # added
            if self.global_mode:
                key_coords = torch.cat([key_coords, global_key_coords], dim=2)  # (N, 3, size+K)

            if self.use_relative_coords:
                key_coords = key_coords - voxel_coords.unsqueeze(-1)
            key_pos_emb = self.k_pos_proj(key_coords)
            key_features = key_features + key_pos_emb

            if self.use_no_query_coords:
                pass
            else:
                query_pos_emb = self.q_pos_proj(voxel_coords).unsqueeze(0)
                query_features = query_features + query_pos_emb

        key_features = key_features.permute(2, 0, 1).contiguous() # (size, N1+N2, C)

        attend_features, attend_weights = self.mhead_attention(
            query = query_features,
            key = key_features,
            value = key_features,
            key_padding_mask = key_mask,
        )

        attend_features = self.drop_out(attend_features)
        voxel_features = voxel_features + attend_features.squeeze(0)
        voxel_features = self.norm1(voxel_features)
        act_features = self.linear2(self.dropout1(self.activation(self.linear1(voxel_features))))
        voxel_features = voxel_features + self.dropout2(act_features)
        voxel_features = self.norm2(voxel_features)
        voxel_features = self.output_layer(voxel_features)
        sp_tensor.features = voxel_features
        return sp_tensor

class AttentionResBlockv2(nn.Module):
    def __init__(self, model_cfg, use_relative_coords = False, use_pooled_feature = False, use_no_query_coords = False):
        super(AttentionResBlockv2, self).__init__()
        sp_cfg = model_cfg.SP_CFGS
        self.sp_attention = SparseAttention3dv2(
            input_channels = sp_cfg.CHANNELS[0],
            output_channels = sp_cfg.CHANNELS[2],
            ff_channels = sp_cfg.CHANNELS[1],
            dropout = sp_cfg.DROPOUT,
            num_heads = sp_cfg.NUM_HEADS,
            attention_modes = sp_cfg.ATTENTION,
            strides = sp_cfg.STRIDE,
            num_ds_voxels = sp_cfg.NUM_DS_VOXELS,
            use_relative_coords = use_relative_coords,
            use_pooled_feature = use_pooled_feature,
            use_no_query_coords= use_no_query_coords,
            global_mode=sp_cfg.get('GLOBAL_MODE', None),
        )
        subm_cfg = model_cfg.SUBM_CFGS
        self.subm_attention_modules = nn.ModuleList()
        for i in range(subm_cfg.NUM_BLOCKS):
            self.subm_attention_modules.append(SubMAttention3dv2(
                input_channels = subm_cfg.CHANNELS[0],
                output_channels = subm_cfg.CHANNELS[2],
                ff_channels = subm_cfg.CHANNELS[1],
                dropout = subm_cfg.DROPOUT,
                num_heads = subm_cfg.NUM_HEADS,
                attention_modes = subm_cfg.ATTENTION,
                use_pos_emb =  subm_cfg.USE_POS_EMB,
                use_relative_coords = use_relative_coords,
                use_no_query_coords= use_no_query_coords,
                global_mode=subm_cfg.get('GLOBAL_MODE', None),
            ))

    def forward(self, sp_tensor):
        sp_tensor = self.sp_attention(sp_tensor)
        indentity_features = sp_tensor.features
        for subm_module in self.subm_attention_modules:
            sp_tensor = subm_module(sp_tensor)
        sp_tensor.features += indentity_features
        return sp_tensor

class VoxelTransformerV2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range):
        super(VoxelTransformerV2, self).__init__()
        self.model_cfg = model_cfg

        self.use_relative_coords = self.model_cfg.get('USE_RELATIVE_COORDS', False)
        self.use_pooled_feature = self.model_cfg.get('USE_POOLED_FEATURE', False)
        self.use_no_query_coords = self.model_cfg.get('USE_NO_QUERY_COORDS', False)

        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.input_transform = nn.Sequential(
            nn.Linear(input_channels, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.backbone = nn.ModuleList()
        for param in self.model_cfg.PARAMS:
            self.backbone.append(AttentionResBlockv2(param, self.use_relative_coords, self.use_pooled_feature, self.use_no_query_coords))

        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        voxel_features = self.input_transform(voxel_features)

        sp_tensor = SparseTensor(
            features = voxel_features,
            indices = voxel_coords.int(),
            spatial_shape = self.grid_size,
            voxel_size = self.voxel_size,
            point_cloud_range = self.point_cloud_range,
            batch_size = batch_size,
            hash_size = self.model_cfg.HASH_SIZE,
            map_table = None,
            gather_dict = None,
        )
        for attention_block in self.backbone:
            sp_tensor = attention_block(sp_tensor)

        batch_dict.update({
            'encoded_spconv_tensor': sp_tensor,
            'encoded_spconv_tensor_stride': 8
        })
        return batch_dict

class SparseConvTensor(object):
    def __init__(self, features, indices):
        self.features = features
        self.indices = indices

class VoxelTransformerV3(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range):
        super(VoxelTransformerV3, self).__init__()
        self.model_cfg = model_cfg

        self.use_relative_coords = self.model_cfg.get('USE_RELATIVE_COORDS', False)
        self.use_pooled_feature = self.model_cfg.get('USE_POOLED_FEATURE', False)
        self.use_no_query_coords = self.model_cfg.get('USE_NO_QUERY_COORDS', False)

        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.input_transform = nn.Sequential(
            nn.Linear(input_channels, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.backbone = nn.ModuleList()
        for param in self.model_cfg.PARAMS:
            self.backbone.append(AttentionResBlock(param, self.use_relative_coords, self.use_pooled_feature, self.use_no_query_coords))

        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        voxel_features = self.input_transform(voxel_features)

        x_convs = [SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
        )]

        sp_tensor = SparseTensor(
            features = voxel_features,
            indices = voxel_coords.int(),
            spatial_shape = self.grid_size,
            voxel_size = self.voxel_size,
            point_cloud_range = self.point_cloud_range,
            batch_size = batch_size,
            hash_size = self.model_cfg.HASH_SIZE,
            map_table = None,
            gather_dict = None,
        )
        for attention_block in self.backbone:
            sp_tensor = attention_block(sp_tensor)
            x_convs.append(SparseConvTensor(
                features=sp_tensor.features,
                indices=sp_tensor.indices,
            ))

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_convs[0],
                'x_conv2': x_convs[1],
                'x_conv3': x_convs[2],
                'x_conv4': x_convs[3],
            }
        })

        batch_dict.update({
            'encoded_spconv_tensor': sp_tensor,
            'encoded_spconv_tensor_stride': 8
        })
        return batch_dict