import torch
from argparse import ArgumentParser
from mmengine import Config, DictAction, mkdir_or_exist

def arg_parse():
    parser = ArgumentParser('Edit model weights.')
    parser.add_argument('config', help='Path to the config file.')
    parser.add_argument(
        '--edit_concepts', 
        '-e',
        help="Concepts to edit. You can enter multiple concepts, separated by ','.  "
    )
    parser.add_argument(
        '--preserve_concepts',
        '-p',
        help="Concepts to preserve. You can enter multiple concepts, seperated by ','"
    )
    parser.add_argument(
        'guide_concepts',
        '-g',
        help = "what concepts to guide the edit to. You can enter multiple concepts, seperated by ','"
    )
    parser.add_argument(
        '--work-dir',
        '-w',
        default='./workdirs/default/',
        help='Directory to save the output files.')
    parser.add_argument(
        '--extra_concepts',
        '-x',
        choices = ['object'],
        help = "Extra prompt for the edit. If it is 'object', then extra prompts are like 'image of {concept}'. Otherwise, no extra concepts will be used."
    )
    parser.add_argument(
        '--editor-ckpt',
        '-c',
        help = "Checkpoint of an continuation of an editor. If it is not given then it will edit the pre-trained model weight."
    )
    parser.add_argument(   
        '--gpu',
        '-g',
        default = 0,
        help = "GPU to use"
    )
    parser.add_argument(
        '--cfg-files',
        '-c',
        help = "The different config files to use"
    )

    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    work_dir = args.work_dir
    mkdir_or_exist(work_dir)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    cfg = Config.fromfile(args.config)
    cfg.editor.update({'device': device})
    if args.cfg_files:
        cfg.merge_from_dict(args.cfg_files) #Update the config file with the new config file

    # editor = EDITORS.build(cfg.editor)

    edit_concepts, preserve_concepts, guide_concepts = arg_parse(
        args.edit_concepts, args.preserve_concepts, args.guide_concepts, with_extra=args.extra_concepts
    )

    # editor.edit(edit_concepts, guided_concepts, preserve_concepts)
    # save_editor(editor, osp.join(work_dir, f'editor_{editor.id_code}.pt'))
    
    if __name__ == '__main__':
        main()  

    
