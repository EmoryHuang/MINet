import argparse


def get_pretraining_args():
    parser = argparse.ArgumentParser(description="Pretraining arguments.")
    parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use.')
    parser.add_argument('--mode', default='train', type=str, help='train/test')
    parser.add_argument('--pretrain_model',
                        default='distilbert-base-uncased',
                        type=str,
                        help='Pretrain model.')
    parser.add_argument('--dataset',
                        default='/home/hby/POI/Datasets/ABSA/ABSA_data.txt',
                        type=str,
                        help='Raw dataset.')
    parser.add_argument('--model_dir', default='./Model', type=str, help='Model dir.')
    parser.add_argument('--batch_size', default=256, help='Batch size.')
    parser.add_argument('--learning_rate',
                        default=0.000001,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('--epochs', default=50, type=int, help='Train epochs.')

    args = parser.parse_args()
    return args


# def get_KGEmbedding_args():
#     parser = argparse.ArgumentParser(description="KGEmbedding arguments.")
#     parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use.')
#     parser.add_argument('--mode', default='train', type=str, help='train/test')
#     parser.add_argument('--model_type',
#                         default='distmult',
#                         type=str,
#                         help='distmult/transe/complex.')
#     parser.add_argument('--dataset',
#                         default='Yelp',
#                         type=str,
#                         help='Yelp/Gowalla/Foursquare')
#     parser.add_argument('--database',
#                         default='./Datasets',
#                         type=str,
#                         help='Database dir.')
#     parser.add_argument('--model_dir', default='./Model', type=str, help='Model dir.')
#     parser.add_argument('--batch_size', default=1024, help='Batch size.')
#     parser.add_argument('--hidden_size', default=128, type=int, help='Hidden size.')
#     parser.add_argument('--learning_rate',
#                         default=0.0001,
#                         type=float,
#                         help='Learning rate.')
#     parser.add_argument('--epochs', default=100, type=int, help='Train epochs.')

#     args = parser.parse_args()

#     if args.dataset == 'Yelp':
#         args.checkins_dataset = './Datasets/checkins_30.feather'
#         args.friends_dataset = './Datasets/friends.feather'
#         args.categories_dataset = './Datasets/categories.feather'
#     return args


def get_args():
    parser = argparse.ArgumentParser(description="Training arguments.")
    parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use.')
    parser.add_argument('--mode', default='train', type=str, help='train/test')
    parser.add_argument('--dataset',
                        default='yelp',
                        type=str,
                        help='yelp/gowalla/foursquare')
    parser.add_argument('--database',
                        default='./Datasets',
                        type=str,
                        help='Database dir.')
    parser.add_argument('--max_sequence_length',
                        default=15,
                        type=int,
                        help='Max sequence length.')
    parser.add_argument('--model_path',
                        default='./Model/model_yelp.pkl',
                        type=str,
                        help='model path')
    parser.add_argument('--model_dir', default='./Model', type=str, help='Model dir.')
    parser.add_argument('--tkg_batch_size', default=512, type=int, help='TKG batch size.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--hidden_size', default=64, type=int, help='Hidden size.')
    parser.add_argument('--learning_rate',
                        default=0.001,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('--epochs', default=50, type=int, help='Train epochs.')
    parser.add_argument('--use_absa', default=True, type=bool, help='Add absa')
    parser.add_argument('--use_social', default=True, type=bool, help='Add social')
    parser.add_argument('--use_cate', default=True, type=bool, help='Add category')
    parser.add_argument('--lamb', default=0.01, type=float)
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--group', default='month', type=str)

    args = parser.parse_args()

    if args.dataset == 'yelp':
        args.checkins_dataset = './Datasets/checkins_30.feather'
        args.friends_dataset = './Datasets/friends.feather'
        args.categories_dataset = './Datasets/categories.feather'
    elif args.dataset == 'yelp-wo':
        args.checkins_dataset = './Datasets/checkins_30.feather'
        args.friends_dataset = './Datasets/friends.feather'
        args.categories_dataset = './Datasets/categories.feather'
        args.use_absa = False
        args.use_social = True
        args.use_cate = False
    elif args.dataset == 'gowalla':
        args.checkins_dataset = './Datasets/checkins_gowalla.feather'
        args.friends_dataset = './Datasets/friends_gowalla.feather'
        args.use_absa = False
        args.use_social = True
        args.use_cate = False
    elif args.dataset == 'foursquare':
        args.checkins_dataset = './Datasets/checkins_foursquare.feather'
        args.friends_dataset = './Datasets/friends_foursquare.feather'
        args.use_absa = False
        args.use_social = True
        args.use_cate = False

    return args