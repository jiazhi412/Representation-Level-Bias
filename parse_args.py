import os
import argparse
import torch
import utils
import model.MINE as MINE
import model.RLB as RLB
import model.Logits_Loss as Logits_Loss

def collect_args():
    parser = argparse.ArgumentParser(description='Calculate Representation-Level Bias')
    # experiment choices
    parser.add_argument('--experiment', metavar='', choices=[
                                 # FairFace
                                 'FairFace_attr_baseline_model',
                                 'FairFace_attr_baseline',
                                 'FairFace_attr_weighting',
                                 'FairFace_attr_domain_adaption',
                                 'FairFace_attr_domain_independent',
                                 'FairFace_attr_representation_disentanglement_uniconf',
                                 'FairFace_attr_representation_disentanglement_gradproj',

                                 # StyleGan2
                                 'StyleGan2_baseline',

                                 # Colored_MNIST
                                 'colored_model',
                                 'colored_data',

                                 # Celeba
                                 'celeba_baseline',
                                 'celeba_weighting',
                                 'celeba_domain_adaption',
                                 'celeba_domain_independent',
                                 'celeba_representation_disentanglement_uniconf',
                                 'celeba_representation_disentanglement_gradproj',
                                 
                                 # Ablation study
                                 'UAI', 
                                 'CAI'], default='colored_model')
    parser.add_argument('--experiment-name', type=str, default='debug',
                        help='specifies a name to this experiment for saving the model and result)')
    parser.add_argument('-m', '--model', metavar='', choices=['MINE', 'RLB', 'LL'], default='RLB',
                        help='bias assessment model name')
    parser.add_argument('-b', '--bias', metavar='', choices=['gender', 'race', 'label', 'color'],
                        help='bias name')

    ## only for FairFace (female percentage or black race percentage)
    parser.add_argument('--percentage', type=float, default=None) 
    parser.add_argument('--more-race', action='store_true') 
    ## only for colored MNIST (color standard deviation)
    parser.add_argument('--color-std', type=float, default=None) 
    ## only for celeba (synthesized representations)
    parser.add_argument('--attack', type=str, choices=['shuffleR', 'fakeR', 'shuffleZ', 'fakeZ'], default=None)
    ## only for styleGan2 (sex entropy or race entropy)
    parser.add_argument('--entropy', type=float, default=None)
    ## only for UAI and CAI
    parser.add_argument('-f', '--feature', metavar='', choices=['e1', 'e2', 'h'], default=None, help='feature name')
    parser.add_argument('-d', '--dataset', metavar='', choices=['Adult', 'YaleB', 'German'], default=None, help='dataset name')

    # experiment settings
    parser.add_argument('-l', '--label-mode', metavar='', choices=['category', 'onehot'], default='category', help='label mode')
    parser.add_argument('-p', '--preprocessing', metavar='', choices=['normal', 'indices'], default='normal', help='preprocessing mode')
    parser.add_argument('-s', '--sample', metavar='', choices=['same', 'different'], default='same', help='sample mode')
    parser.add_argument('-i', '--indices', type=int, default=1)

    # hyper-parameter
    parser.add_argument('--iter-num', type=int, metavar='', default=40000)
    parser.add_argument('--window-size', type=int, metavar='', default=1000)
    parser.add_argument('--ma_et',type=float, metavar='', default=1.0)
    parser.add_argument('--encoder-dim', type=int, metavar='', default=10)
    parser.add_argument('--batch-size', type=int, metavar='', default=1000)
    parser.add_argument('--hidden-dim', type=int, metavar='', default=1000)
    parser.add_argument('--learning-rate', type=float, metavar='', default=1e-3)
    parser.add_argument('--converge-criterion', action='store_true') 
    parser.add_argument('--stop-error', type=float, metavar='', default=1e-7)
    parser.add_argument('--stop-region', type=float, metavar='', default=2000)

    # running settings
    parser.add_argument('--with-cuda', dest='cuda', action='store_true')
    parser.add_argument('--random-seed', type=int, default=26)
    parser.add_argument('-sm', '--save-model', action='store_true', help='save model')

    opt = vars(parser.parse_args())
    opt, model, optim, data = create_experiment_setting(opt)
    return opt, model, optim, data

def create_experiment_setting(opt):
    opt['device'] = torch.device('cuda' if opt['cuda'] and torch.cuda.is_available() else 'cpu')
    utils.set_random_seed(opt['random_seed'])
    opt['save_folder'] = os.path.join('result', opt['experiment'], opt['experiment_name'])
    utils.create_folder(opt['save_folder'])
    print(opt)
    
    # Load data (r -> learned representations and z -> protected attributed)
    if opt['experiment'].startswith('FairFace'):
        opt['save_path'] = os.path.join(opt['save_folder'], '_'.join([opt['experiment'], opt['bias'], str(opt['percentage']), opt['label_mode']]))
        r, z = utils.load_FairFace(opt, opt['experiment'], opt['percentage'], protected_attribute=opt['bias'], onehot_or_category=opt['label_mode'])

    elif opt['experiment'].startswith('colored'):
        opt['save_path'] = os.path.join(opt['save_folder'], '_'.join([opt['experiment'], opt['bias'], str(opt['color_std']), opt['label_mode']]))
        r, z = utils.load_coloredMNIST(opt['color_std'], opt['experiment'], label_or_bias=opt['bias'], onehot_or_category=opt['label_mode'])

    elif opt['experiment'].startswith('celeba'):
        opt['save_path'] = os.path.join(opt['save_folder'], '_'.join([opt['experiment'], opt['bias'], opt['label_mode']]))
        r, z = utils.load_celeba(opt, opt['experiment'], label_or_bias=opt['bias'], onehot_or_category=opt['label_mode'])
        if opt['attack'] != None:
            r, z = utils.attack_generator(opt['attack'], r, z)

    elif opt['experiment'].startswith('StyleGan2'):
        opt['save_path'] = os.path.join(opt['save_folder'], '_'.join([opt['experiment'], opt['bias'], str(opt['entropy']), opt['label_mode']]))
        r, z = utils.load_StyleGan2(opt, opt['experiment'], label_or_bias=opt['bias'], onehot_or_category=opt['encoder'])

    elif opt['experiment'] in ['UAI', 'CAI']:
        opt['save_folder'] = os.path.join('result', opt['experiment'], opt['experiment_name'], opt['dataset'], opt['feature'])
        utils.create_folder(opt['save_folder'])
        opt['save_path'] = os.path.join(opt['save_folder'], '_'.join([opt['experiment'], opt['bias']], opt['label_mode'], opt['preprocessing'], opt['sample']))
        r, z = utils.get_r_z(model_name=opt['experiment'], dataset_name=opt['dataset'], feature_name=opt['feature'],
                            bias_name=opt['bias'], onehot_or_category=opt['encoder'])

    # Load model
    if opt['model'] == 'MINE':
        net = MINE.Mine(input_size=r.shape[1] + z.shape[1], hidden_size=opt['hidden_dim']).to(opt['device'])
        optim = torch.optim.Adam(net.parameters(), lr=opt['learning_rate'])
    elif opt['model'] == 'RLB':
        net = RLB.RLB(r.shape[1], z.shape[1], opt)
        optim = torch.optim.Adam(net.parameters(), lr=opt['learning_rate'])
    elif opt['model'] == 'LL':
        net = Logits_Loss.LL(r.shape[1], 2)
        optim = torch.optim.Adam(net.parameters(), lr=opt['learning_rate'])

    return opt, net.to(opt['device']), optim, (r.float().to(opt['device']), z.float().to(opt['device']))
