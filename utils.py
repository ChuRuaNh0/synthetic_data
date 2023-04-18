import cv2

def load_logo_sublogo_id(file_info):
    # Load info file
    lines = open(file_info).read().strip('\n').split('\n')
    dict_logo = dict()
    for line in lines:
        tmp = line.strip().split(',')
        if len(tmp) == 3:
            assert tmp[0] == 'logo', "Must be 'logo' category: {}".format(line)
            logo_img = cv2.imread(tmp[1], cv2.IMREAD_UNCHANGED)
            id  = int(tmp[2])
            assert tmp[1] not in dict_logo, "{} already exits".format(tmp[1])
            dict_logo[tmp[1]] = {'logo': logo_img, 'id': id}
        elif len(tmp) == 4:
            assert tmp[0] == 'sublogo', "Must be 'sublogo' category: {}".format(line)
            sublogo_img = cv2.imread(tmp[1], cv2.IMREAD_UNCHANGED)
            id  = int(tmp[3])
            assert tmp[2] in dict_logo, "{} not exits".format(tmp[2])
            dict_logo[tmp[2]]['sublogo'] = sublogo_img
            dict_logo[tmp[2]]['sublogo_id'] = id
        else:
            raise NotImplementedError("Please read document carefully ...")
    return dict_logo

def load_and_show_meta(file_info):
    dict_logo = load_logo_sublogo_id(file_info)
    for logo in dict_logo:
        if 'sublogo' in dict_logo[logo]:
            print('- ',
                logo,
                dict_logo[logo]['logo'].shape,
                dict_logo[logo]['id'],
                dict_logo[logo]['sublogo'].shape,
                dict_logo[logo]['sublogo_id'])
        else:
            print('- ',
                logo,
                dict_logo[logo]['logo'].shape,
                dict_logo[logo]['id'])
    return dict_logo


if __name__ == '__main__':
    load_and_show_meta('png/info.txt')