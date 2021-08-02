import os

def get_tokenizer(args):
    from transformers.pytorch_transformers import BertTokenizer

    tokenizer_class = BertTokenizer

    if args.vlnbert == 'oscar':
        model_name_or_path = 'Oscar/pretrained_models/base-no-labels/ep_67_588997'
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    elif args.vlnbert == 'vilbert':
        tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', do_lower_case=True)
    return tokenizer

def get_vlnbert_models(args, config=None):
    if args.vlnbert == 'oscar':
        from vlnbert.vlnbert_OSCAR import VLNBert
        from transformers.pytorch_transformers import BertConfig

        model_class = VLNBert
        model_name_or_path = 'Oscar/pretrained_models/base-no-labels/ep_67_588997'
        vis_config = BertConfig.from_pretrained(model_name_or_path, num_labels=2, finetuning_task='vln-r2r')

        vis_config.model_type = 'visual'
        vis_config.finetuning_task = 'vln-r2r'
        vis_config.hidden_dropout_prob = 0.3
        vis_config.hidden_size = 768
        vis_config.img_feature_dim = 2176
        vis_config.num_attention_heads = 12
        vis_config.num_hidden_layers = 12
        visual_model = model_class.from_pretrained(model_name_or_path, from_tf=False, config=vis_config)

    elif args.vlnbert == 'vilbert':
        from vlnbert.vlnbert_CA import VLNBert
        from vlnbert.vlnbert_CA import BertConfig

        # '/data2/csz/VLN/R2R/released/vln-bert/run_220825_pytorch_model_10.bin'
        model_name_or_path = args.init_bert_file

        vis_config = BertConfig.from_json_file(os.path.join(
            'snap/vln-bert', 
            'config/bert_base_6_layer_6_connect.json'))
        vis_config.img_feature_dim = 2048 + args.angle_feat_size
        vis_config.img_feature_type = args.features
        vis_config.layer_norm_eps = 1e-12
        vis_config.hidden_dropout_prob = 0.3
        vis_config.v_hidden_dropout_prob = 0.3
        
        if model_name_or_path:
            visual_model = VLNBert.from_pretrained(model_name_or_path, config=vis_config)
        else:
            visual_model = VLNBert(vis_config)

    return visual_model
