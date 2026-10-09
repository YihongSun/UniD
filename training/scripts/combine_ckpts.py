import os
import os.path as osp
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Combine multiple model checkpoints into one directory via soft link.")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Paths to the distilled model checkpoint directories.")
    parser.add_argument("--base_model", type=str, default="checkpoints/pretrained_sd2", help="Local Stable Diffusion base-model directory used for model_index.json.")
    return parser

# python3 scripts/combine_ckpts.py -c 'model-finetuned/sd2_uni_distill_MemAttnTemp(con)(dec)_B48_T4_50000_dev_256x2'
# python3 scripts/combine_ckpts.py -c 'model-finetuned/sd2_uni_distill_MemAttnTemp(con)(dec)_B48_T4_50000_dev_256x2/checkpoint-50000'

if "__main__" == __name__:
    args = get_parser().parse_args()
    ckpt_path = osp.join(os.getcwd(), args.ckpt.rstrip('/'))
    if not osp.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}. Pass a local checkpoint with --ckpt.")
    contents = os.listdir(ckpt_path)
    assert 'unet' in contents, f'unet/ should be in {ckpt_path}.'
    assert 'connectors' in contents, f'unet/ should be in {ckpt_path}.'
    unet_path = osp.join(ckpt_path, 'unet')
    connectors_path = osp.join(ckpt_path, 'connectors')
    if 'arguments.txt' not in contents:
        arg_path = osp.join(osp.dirname(ckpt_path), 'arguments.txt')
        model_index_path = osp.join(osp.dirname(ckpt_path), 'model_index.json')
        ckpt_idx = int(osp.basename(ckpt_path).split('-')[-1])
        ckpt_name = osp.basename(osp.dirname(ckpt_path))
        ckpt_name_ls = ckpt_name.split('_')
        assert int(ckpt_name_ls[6]) % 1000 == 0, "The original checkpoint should be a multiple of 1000."
        ckpt_name_ls[6] = str(ckpt_idx)
        ckpt_name = '_'.join(ckpt_name_ls) + '_combined'
        ckpt_dir = osp.dirname(osp.dirname(ckpt_path))
    else:
        arg_path = osp.join(ckpt_path, 'arguments.txt')
        model_index_path = osp.join(ckpt_path, 'model_index.json')
        ckpt_name = osp.basename(ckpt_path) + '_combined'
        ckpt_dir = osp.dirname(ckpt_path)
    assert osp.exists(arg_path), f'arguments.txt should be found'
    if not osp.exists(model_index_path):
        model_index_path = osp.join(args.base_model, "model_index.json")
    if not osp.exists(model_index_path):
        raise FileNotFoundError(
            f"model_index.json not found beside checkpoint or in base model '{args.base_model}'. "
            "Provide the local Stable Diffusion base model with --base_model."
        )
    
    out_ckpt_path = osp.join(ckpt_dir, ckpt_name)
    os.makedirs(out_ckpt_path, exist_ok=True)
    os.makedirs(osp.join(out_ckpt_path, 'projectors'), exist_ok=True)

    with open(arg_path, "r") as f:
        arguments = [l.strip() for l in f.readlines()]
        arg = eval([l for l in arguments if l.startswith("_content:")][0].replace("_content:", "").strip())
    
    for dtask, conf in arg['distill_tasks'].items():
        if conf['ckpt'] is None:    # handle cases where no ckpt is provided (can directly use input latent as target)
            if dtask == "restore":
                conf['ckpt'] = "checkpoints/sd2_specialist_R_placeholder"
            else:
                raise ValueError(f"ckpt cannot be None for task {dtask}.")
            
        uni_ckpt_path = osp.join(os.getcwd(), conf['ckpt'].rstrip('/'))
        uni_contents = os.listdir(uni_ckpt_path)
        assert 'projectors' in uni_contents, f'projectors/ should be in {uni_ckpt_path}.'
        projectors_path = osp.join(uni_ckpt_path, 'projectors')
        os.system(f'cp "{projectors_path}/"*.bin "{out_ckpt_path}/projectors/"')
        # Need to add projectors loaded from this ckpt into arguments.txt
        arg_txt = osp.join(uni_ckpt_path, 'arguments.txt')
        if not osp.exists(arg_txt):
            arg_txt = osp.join(osp.dirname(uni_ckpt_path), 'arguments.txt')
        with open(arg_txt, "r") as f:
            arguments_ = [l.strip() for l in f.readlines()]
            arg_ = eval([l for l in arguments_ if l.startswith("_content:")][0].replace("_content:", "").strip())

        ## V2 compatibility
        if type(arg_['projectors']) is str:
            arg['projectors'] = arg_['projectors']
            if 'all_projectors' not in arg:
                arg['all_projectors'] = dict()
            if 'task_to_train' not in arg:
                arg['task_to_train'] = list()

            task = arg_['task_to_train']
            arg['all_projectors'][task] = arg_['all_projectors'][task]
            arg['task_to_train'].append(task)
            continue

        ## V1 compatibility
        if 'projectors' not in arg:
            arg['projectors'] = arg_['projectors']
        else:
            assert arg['projectors']['hidden_dims'] == arg_['projectors']['hidden_dims']
            arg['projectors']['heads'].update(arg_['projectors']['heads'])

    os.system(f'cp -r "{unet_path}" "{out_ckpt_path}/unet"')
    os.system(f'cp -r "{connectors_path}" "{out_ckpt_path}/connectors"')
    os.system(f'cp "{model_index_path}" "{out_ckpt_path}/model_index.json"')
    with open(osp.join(out_ckpt_path, "arguments.txt"), "w") as f:
        for l in arguments:
            if l.startswith("_content:"):
                f.write("_content: " + str(arg) + "\n")
            else:
                f.write(l + "\n")
