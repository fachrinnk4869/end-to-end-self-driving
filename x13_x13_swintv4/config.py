import os


class GlobalConfig:
    gpu_id = '0'
    model = 'x13_x13_swintv4'
    logdir = 'log/'+model  # +'_w1' for 1 weather only
    init_stop_counter = 15

    n_class = 23
    batch_size = 8
    coverage_area = 64

    # MGN parameter
    MGN = True
    loss_weights = [1, 1, 1, 1, 1, 1, 1]
    lw_alpha = 1.5
    bottleneck = [335, 756]

    # for Data
    seq_len = 1  # jumlah input seq
    pred_len = 3  # future waypoints predicted

    # 14_weathers_full_data OR clear_noon_full_data
    root_dir = '/media/fachri/banyak/endtoend/data/ADVERSARIAL/ClearNoon-fix'
    # root_dir = '/media/fachri/banyak/endtoend/data/ADVERSARIAL/COBA'
    train_towns = [
        'Town01', 'Town02', 'Town03', 'Town04',
        'Town06', 'Town07',
        'Town10'
    ]
    val_towns = ['Town05']
    # val_towns = ['Town10']
    train_data, val_data = [], []
    for town in train_towns:
        if not (town == 'Town07' or town == 'Town10'):
            train_data.append(os.path.join(root_dir, town+'_long'))
        train_data.append(os.path.join(root_dir, town+'_short'))
        train_data.append(os.path.join(root_dir, town+'_tiny'))
    for town in val_towns:
        val_data.append(os.path.join(root_dir, town+'_short'))
        val_data.append(os.path.join(root_dir, town+'_tiny'))

    # buat prediksi expert, test
    test_data = []
    test_weather = 'ClearNoon-fix'  # ClearNoon
    test_scenario = 'ADVERSARIAL'  # NORMAL ADVERSARIAL
    expert_dir = '/media/fachri/banyak/endtoend/data/' + \
        test_scenario+'/'+test_weather  # 8T1W 8T14W
    for town in val_towns:
        # Expert OR Expert_w1 for 1 weather only scenario
        test_data.append(os.path.join(expert_dir, town+'_long'))

    input_resolution = 224
    scale = 1  # image pre-processing
    crop = 224  # image pre-processing
    res_resize = crop  # image pre-processing
    lr = 1e-4  # learning rate AdamW
    weight_decay = 1e-3

    # Controller
    # control weights untuk PID dan MLP dari tuningan MGN
    # urutan steer, throttle, brake
    # baca dulu trainval_log.csv setelah training selesai, dan normalize bobotnya 0-1
    # LWS: lw_wp lw_str lw_thr lw_brk saat convergence
    lws = [0.999884128570556, 1.00005769729614,
           0.999974191188812, 1.00002384185791]
    # : [0.999923467636108, 1.00002813339233, 0.999968945980072, 0.999997556209564]
    # _w1 : [0.99992048740387, 1.00002562999725, 0.999992907047272, 1.00002014636993]
    # _t2 : [0.999884128570556, 1.00005769729614, 0.999974191188812, 1.00002384185791]
    # _t2w1 : [0.9998899102211, 1.0000467300415, 0.999997615814209, 1.00002729892731]
    cw_pid = [lws[0]/(lws[0]+lws[1]), lws[0]/(lws[0]+lws[2]),
              lws[0]/(lws[0]+lws[3])]  # str, thrt, brk
    cw_mlp = [1-cw_pid[0], 1-cw_pid[1], 1-cw_pid[2]]  # str, thrt, brk

    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.4  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25  # maximum change in speed input to logitudinal controller
    min_act_thrt = 0.2  # minimum throttle

    # ORDER DALAM RGB!!!!!!!!
    SEG_CLASSES = {
        'colors': [[0, 0, 0], [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60],
                   [153, 153, 153], [157, 234, 50], [
            128, 64, 128], [244, 35, 232], [107, 142, 35],
            [0, 0, 142], [102, 102, 156], [
            220, 220, 0], [70, 130, 180], [81, 0, 81],
            [150, 100, 100], [230, 150, 140], [
            180, 165, 180], [250, 170, 30], [110, 190, 160],
            [170, 120, 50], [45, 60, 150], [145, 170, 100]],
        'classes': ['None', 'Building', 'Fences', 'Other', 'Pedestrian',
                    'Pole', 'RoadLines', 'Road', 'Sidewalk', 'Vegetation',
                            'Vehicle', 'Wall', 'TrafficSign', 'Sky', 'Ground',
                            'Bridge', 'RailTrack', 'GuardRail', 'TrafficLight', 'Static',
                            'Dynamic', 'Water', 'Terrain']
    }
    n_vit_b16 = [[32, 24], [32], [64], [88, 128], [208, 352, 512], [150]]
    n_fmap_b1 = [[32, 16], [24], [40], [80, 112], [192, 320, 1152]]
    n_fmap_b3 = [[40, 24], [32], [48], [96, 136], [232, 384, 1536]]
    n_decoder = n_vit_b16

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
