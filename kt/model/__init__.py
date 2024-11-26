def get_model(model_config):
    print(model_config)
    if model_config['model'] == 'AKT':
        from .akt import AKT
        return AKT(s_num=model_config['s_num'],
                   q_num=model_config['q_num'],
                   n_blocks=1,
                   d_model=256,
                   dropout=0.05,
                   kq_same=1,
                   model_type='akt',
                   l2=1e-5
                   )
    raise NotImplementedError(f"Model {model_config['model']} Not Supported Yet")
