import pandas as pd
import utils

checkins_path = '/home/hby/POI/Datasets/Yelp/yelp_academic_dataset_review.json'
poi_path = '/home/hby/POI/Datasets/Yelp/yelp_academic_dataset_business.json'
users_path = '/home/hby/POI/Datasets/Yelp/yelp_academic_dataset_user.json'


def user_filter(checkins, thr=20):
    # filter users who visited less than `thr` POIs and POIs visited by less than `thr` users
    def check(checkins):
        if not (checkins.groupby('user_id').count() > thr).all(axis=None):
            return False
        if not (checkins.groupby('business_id').count() > thr).all(axis=None):
            return False
        return True

    while not check(checkins):
        group_user = checkins.groupby('user_id').count()
        user = group_user.loc[group_user['date'] > thr].index.values
        checkins = checkins.loc[checkins['user_id'].isin(user)]

        group_loc = checkins.groupby('business_id').count()
        loc = group_loc.loc[group_loc['date'] > thr].index.values
        checkins = checkins.loc[checkins['business_id'].isin(loc)]
    checkins.reset_index(drop=True, inplace=True)
    return checkins


def item2id(checkins, column):
    # id start from 1
    item = checkins[column].unique()
    item2id = dict(zip(item, range(1, item.size + 1)))
    checkins[column] = checkins[column].map(item2id)
    return checkins, item2id


def generate_checkins(filter=20):
    print(f'load checkins data from {poi_path}')
    poi = pd.read_json(poi_path, lines=True)
    poi = poi[['business_id', 'latitude', 'longitude', 'categories']]
    poi = poi.dropna().drop_duplicates()
    poi.reset_index(drop=True, inplace=True)

    checkins = pd.read_json(checkins_path, lines=True)
    checkins = checkins[['user_id', 'business_id', 'text', 'date']]
    checkins = checkins.dropna().drop_duplicates()
    checkins.reset_index(drop=True, inplace=True)
    print(len(checkins))

    # filter
    if filter != 'full' and isinstance(filter, int):
        print(f'checkins data filter with thr {filter}')
        checkins = user_filter(checkins, filter)
        print(len(checkins))

    checkins = pd.merge(checkins, poi, on='business_id')
    checkins = checkins[[
        'user_id', 'business_id', 'latitude', 'longitude', 'date', 'text'
    ]]
    checkins = checkins.dropna()
    checkins = checkins.sort_values(['user_id', 'date'])
    checkins = checkins[checkins['date'].dt.year >= 2007]
    checkins.reset_index(drop=True, inplace=True)

    print(f'change to id')
    checkins, user2id = item2id(checkins, 'user_id')
    checkins, poi2id = item2id(checkins, 'business_id')
    checkins = utils.geohash_encode(checkins)
    checkins, geo2id = item2id(checkins, 'geohash')
    checkins.columns = [
        'user', 'location', 'latitude', 'longitude', 'time', 'text', 'geohash'
    ]
    print('saving checkins data')
    checkins.to_feather(f'./Datasets/checkins_{filter}.feather')
    return user2id, poi2id


def generate_friends(user2id):
    print(f'load users data from {users_path}')
    users = pd.read_json(users_path, lines=True)
    users = users[['user_id', 'friends']]
    users = users.dropna().drop_duplicates()
    users.reset_index(drop=True, inplace=True)

    users['user_id'] = users['user_id'].map(user2id, na_action='ignore')
    users = users[users['friends'] != 'None'].dropna()

    def friends2id(df):
        user_id, friends = df
        friends = [c.strip() for c in friends.split(',')]
        friends = [user2id[c] for c in friends if c in user2id.keys()]
        for friend in friends:
            friends_dict.append([int(user_id), friend])

    print('social relationship analysis')
    friends_dict = []
    _ = users.apply(friends2id, axis=1)
    friends = pd.DataFrame(friends_dict, columns=['user', 'friend'])
    friends = friends.sort_values(['user', 'friend'])
    friends.reset_index(drop=True, inplace=True)

    print('saving social data')
    friends.to_feather('./Datasets/friends.feather')


def generate_categories(poi2id):
    print(f'load categories data from {poi_path}')
    poi = pd.read_json(poi_path, lines=True)
    poi = poi[['business_id', 'categories']]
    poi = poi.dropna().drop_duplicates()
    poi.reset_index(drop=True, inplace=True)

    poi['business_id'] = poi['business_id'].map(poi2id, na_action='ignore')
    poi = poi.dropna()

    # cate2id
    categories = []
    for cates in poi['categories'].unique():
        categories += [c.strip() for c in cates.split(',')]

    categories = sorted(list(set(categories)))
    cate2id = dict(zip(categories, range(1, len(categories) + 1)))

    def categories2id(df):
        business_id, categories = df
        categories = [cate2id[c.strip()] for c in categories.split(',')]
        for cate in categories:
            categories_dict.append([int(business_id), cate])

    print('categories analysis')
    categories_dict = []
    _ = poi.apply(categories2id, axis=1)
    categories = pd.DataFrame(categories_dict, columns=['location', 'category'])
    categories = categories.sort_values(['location', 'category'])
    categories.reset_index(drop=True, inplace=True)

    print('saving categories data')
    categories.to_feather('./Datasets/categories.feather')


def load_checkins(dataset, file_checkin, file_firend):

    checkins = pd.read_csv(file_checkin,
                           sep='\t',
                           names=['user', 'time', 'latitude', 'longitude', 'location'])
    checkins['time'] = pd.to_datetime(checkins['time'], errors='coerce')
    checkins = checkins.dropna().drop_duplicates()
    checkins.reset_index(drop=True, inplace=True)
    checkins = utils.geohash_encode(checkins)
    checkins, user2id = item2id(checkins, 'user')
    checkins, _ = item2id(checkins, 'location')
    checkins, _ = item2id(checkins, 'geohash')
    checkins.to_feather(f'./Datasets/checkins_{dataset}.feather')

    friends = pd.read_csv(file_firend, sep='\t', names=['user', 'friend'])
    friends['user'] = friends['user'].map(user2id)
    friends['friend'] = friends['friend'].map(user2id)
    friends.dropna(inplace=True)
    friends['user'] = friends['user'].astype(int)
    friends['friend'] = friends['friend'].astype(int)
    friends.reset_index(inplace=True, drop=True)
    friends.to_feather(f'./Datasets/friends_{dataset}.feather')

    return checkins, friends


if __name__ == '__main__':
    ### yelp
    user2id, poi2id = generate_checkins(filter=30)
    generate_friends(user2id)
    generate_categories(poi2id)

    ### gowalla
    # load_checkins('gowalla', '/home/hby/POI/Datasets/checkins-gowalla.txt',
    #               '/home/hby/POI/poi_rec_v2/Datasets/gowalla_friend.txt')

    ### 4sq
    # load_checkins('foursquare', '/home/hby/POI/Datasets/checkins-4sq.txt',
    #               '/home/hby/POI/poi_rec_v2/Datasets/4sq_friend.txt')
