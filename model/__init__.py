from .segmenter import ETRIS
from loguru import logger


def build_segmenter(args):
    model = ETRIS(args)
    backbone = []
    head = []
    fix = []
    for k, v in model.named_parameters():
        if (k.startswith('backbone') and 'positional_embedding' not in k or 'bridger' in k) and v.requires_grad:
            backbone.append(v)
        elif v.requires_grad:
            head.append(v)
        else:
            fix.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]
    
    n_backbone_parameters = sum(p.numel() for p in backbone)
    logger.info(f'number of updated params (Backbone): {n_backbone_parameters}.')
    n_head_parameters = sum(p.numel() for p in head)
    logger.info(f'number of updated params (Head)    : {n_head_parameters}')
    n_fixed_parameters = sum(p.numel() for p in fix)
    logger.info(f'number of fixed params             : {n_fixed_parameters}')
    return model, param_list
