from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

df = load_boston()
data = np.concatenate([df.data,df.target.reshape(-1,1)],axis=1).astype(np.float32)
print('data.shape',data.shape)  # (506, 13)

feature_names=list(df.feature_names)
feature_names.append('TARGET')
data_df = pd.DataFrame(data, columns=feature_names)
data_df.describe()


data=np.array(data_df)
np.random.shuffle(data)
train_data = data[:int(data.shape[0] * 0.8)]
# print(train_data.dtype)
test_data = data[int(data.shape[0] * 0.8):]
print(f'train_data.shape: {train_data.shape}, test_data.shape: {test_data.shape}')




from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
mlp= MLPRegressor(activation='logistic',max_iter=1000,verbose=True,solver='sgd',momentum=0)
print(mlp)
mlp.fit(train_data[:,:-1],train_data[:,-1])
mlp_y_predict=mlp.predict(train_data[:,:-1])
train_loss = mean_squared_error(train_data[:,-1], mlp_y_predict)
mlp_y_predict=mlp.predict(test_data[:,:-1])
test_loss = mean_squared_error(test_data[:,-1], mlp_y_predict)
print(f'the mse on the train data :{train_loss}, on the test data: {test_loss}')


log1='''Iteration 1, loss = 0.63928253
Iteration 2, loss = 0.61429349
Iteration 3, loss = 0.59722521
Iteration 4, loss = 0.58114872
Iteration 5, loss = 0.56900362
Iteration 6, loss = 0.56057717
Iteration 7, loss = 0.55592505
Iteration 8, loss = 0.54359736
Iteration 9, loss = 0.53683420
Iteration 10, loss = 0.52738101
Iteration 11, loss = 0.52413941
Iteration 12, loss = 0.51660300
Iteration 13, loss = 0.51925151
Iteration 14, loss = 0.52007737
Iteration 15, loss = 0.51435608
Iteration 16, loss = 0.51178393
Iteration 17, loss = 0.51107728
Iteration 18, loss = 0.51014098
Iteration 19, loss = 0.50738528
Iteration 20, loss = 0.50532067
Iteration 21, loss = 0.50349387
Iteration 22, loss = 0.50227718
Iteration 23, loss = 0.50327330
Iteration 24, loss = 0.50442202
Iteration 25, loss = 0.50128403
Iteration 26, loss = 0.49959244
Iteration 27, loss = 0.49480664
Iteration 28, loss = 0.49320209
Iteration 29, loss = 0.49246847
Iteration 30, loss = 0.49156022
Iteration 31, loss = 0.49104387
Iteration 32, loss = 0.48994727
Iteration 33, loss = 0.48924053
Iteration 34, loss = 0.48810801
Iteration 35, loss = 0.48614924
Iteration 36, loss = 0.48522541
Iteration 37, loss = 0.48470890
Iteration 38, loss = 0.48475568
Iteration 39, loss = 0.48430017
Iteration 40, loss = 0.48284685
Iteration 41, loss = 0.48224833
Iteration 42, loss = 0.48208905
Iteration 43, loss = 0.48153145
Iteration 44, loss = 0.47984210
Iteration 45, loss = 0.47890658
Iteration 46, loss = 0.47823056
Iteration 47, loss = 0.47761814
Iteration 48, loss = 0.47717412
Iteration 49, loss = 0.47640641
Iteration 50, loss = 0.47594537
Iteration 51, loss = 0.47515588
Iteration 52, loss = 0.47453226
Iteration 53, loss = 0.47407637
Iteration 54, loss = 0.47347743
Iteration 55, loss = 0.47311968
Iteration 56, loss = 0.47267632
Iteration 57, loss = 0.47209097
Iteration 58, loss = 0.47106841
Iteration 59, loss = 0.47038781
Iteration 60, loss = 0.46998200
Iteration 61, loss = 0.46979729
Iteration 62, loss = 0.46954360
Iteration 63, loss = 0.46882659
Iteration 64, loss = 0.46840576
Iteration 65, loss = 0.46865344
Iteration 66, loss = 0.46836924
Iteration 67, loss = 0.46790616
Iteration 68, loss = 0.46625356
Iteration 69, loss = 0.46486723
Iteration 70, loss = 0.46429193
Iteration 71, loss = 0.46410304
Iteration 72, loss = 0.46334647
Iteration 73, loss = 0.46260648
Iteration 74, loss = 0.46193465
Iteration 75, loss = 0.46094338
Iteration 76, loss = 0.46009075
Iteration 77, loss = 0.45936260
Iteration 78, loss = 0.45882395
Iteration 79, loss = 0.45813479
Iteration 80, loss = 0.45730507
Iteration 81, loss = 0.45683856
Iteration 82, loss = 0.45659432
Iteration 83, loss = 0.45593401
Iteration 84, loss = 0.45601824
Iteration 85, loss = 0.45560893
Iteration 86, loss = 0.45481427
Iteration 87, loss = 0.45515743
Iteration 88, loss = 0.45490299
Iteration 89, loss = 0.45408107
Iteration 90, loss = 0.45304945
Iteration 91, loss = 0.45235569
Iteration 92, loss = 0.45177775
Iteration 93, loss = 0.45043376
Iteration 94, loss = 0.44975830
Iteration 95, loss = 0.44937184
Iteration 96, loss = 0.44900461
Iteration 97, loss = 0.44937815
Iteration 98, loss = 0.44782342
Iteration 99, loss = 0.44696708
Iteration 100, loss = 0.44630882
Iteration 101, loss = 0.44594704
Iteration 102, loss = 0.44525488
Iteration 103, loss = 0.44486129
Iteration 104, loss = 0.44412686
Iteration 105, loss = 0.44371714
Iteration 106, loss = 0.44276206
Iteration 107, loss = 0.44236597
Iteration 108, loss = 0.44155405
Iteration 109, loss = 0.44081895
Iteration 110, loss = 0.44074908
Iteration 111, loss = 0.44038717
Iteration 112, loss = 0.43930011
Iteration 113, loss = 0.43908332
Iteration 114, loss = 0.43808556
Iteration 115, loss = 0.43769478
Iteration 116, loss = 0.43737200
Iteration 117, loss = 0.43668445
Iteration 118, loss = 0.43614465
Iteration 119, loss = 0.43568923
Iteration 120, loss = 0.43521617
Iteration 121, loss = 0.43487923
Iteration 122, loss = 0.43468572
Iteration 123, loss = 0.43359641
Iteration 124, loss = 0.43296376
Iteration 125, loss = 0.43298492
Iteration 126, loss = 0.43302399
Iteration 127, loss = 0.43196831
Iteration 128, loss = 0.43130479
Iteration 129, loss = 0.43017140
Iteration 130, loss = 0.42976106
Iteration 131, loss = 0.42971084
Iteration 132, loss = 0.42962517
Iteration 133, loss = 0.42843250
Iteration 134, loss = 0.42798635
Iteration 135, loss = 0.42783932
Iteration 136, loss = 0.42754597
Iteration 137, loss = 0.42634865
Iteration 138, loss = 0.42577135
Iteration 139, loss = 0.42524179
Iteration 140, loss = 0.42432070
Iteration 141, loss = 0.42387561
Iteration 142, loss = 0.42325167
Iteration 143, loss = 0.42256846
Iteration 144, loss = 0.42196861
Iteration 145, loss = 0.42132129
Iteration 146, loss = 0.42084444
Iteration 147, loss = 0.42044106
Iteration 148, loss = 0.41992089
Iteration 149, loss = 0.41952693
Iteration 150, loss = 0.41913000
Iteration 151, loss = 0.41875685
Iteration 152, loss = 0.41850822
Iteration 153, loss = 0.41790442
Iteration 154, loss = 0.41754070
Iteration 155, loss = 0.41687721
Iteration 156, loss = 0.41641261
Iteration 157, loss = 0.41590232
Iteration 158, loss = 0.41512433
Iteration 159, loss = 0.41471594
Iteration 160, loss = 0.41425396
Iteration 161, loss = 0.41395516
Iteration 162, loss = 0.41367009
Iteration 163, loss = 0.41300301
Iteration 164, loss = 0.41247788
Iteration 165, loss = 0.41216991
Iteration 166, loss = 0.41169860
Iteration 167, loss = 0.41120826
Iteration 168, loss = 0.41094869
Iteration 169, loss = 0.41018128
Iteration 170, loss = 0.40975297
Iteration 171, loss = 0.40928410
Iteration 172, loss = 0.40896034
Iteration 173, loss = 0.40835584
Iteration 174, loss = 0.40775553
Iteration 175, loss = 0.40734278
Iteration 176, loss = 0.40694487
Iteration 177, loss = 0.40641232
Iteration 178, loss = 0.40601050
Iteration 179, loss = 0.40606859
Iteration 180, loss = 0.40590983
Iteration 181, loss = 0.40484577
Iteration 182, loss = 0.40443945
Iteration 183, loss = 0.40398537
Iteration 184, loss = 0.40378294
Iteration 185, loss = 0.40396058
Iteration 186, loss = 0.40293604
Iteration 187, loss = 0.40334090
Iteration 188, loss = 0.40287055
Iteration 189, loss = 0.40294573
Iteration 190, loss = 0.40196121
Iteration 191, loss = 0.40146504
Iteration 192, loss = 0.40099875
Iteration 193, loss = 0.40030138
Iteration 194, loss = 0.40050867
Iteration 195, loss = 0.40001898
Iteration 196, loss = 0.39965636
Iteration 197, loss = 0.39780934
Iteration 198, loss = 0.39673280
Iteration 199, loss = 0.39635706
Iteration 200, loss = 0.39566596
Iteration 201, loss = 0.39524469
Iteration 202, loss = 0.39502083
Iteration 203, loss = 0.39451982
Iteration 204, loss = 0.39420833
Iteration 205, loss = 0.39379547
Iteration 206, loss = 0.39318903
Iteration 207, loss = 0.39277793
Iteration 208, loss = 0.39261184
Iteration 209, loss = 0.39202216
Iteration 210, loss = 0.39182002
Iteration 211, loss = 0.39140149
Iteration 212, loss = 0.39109846
Iteration 213, loss = 0.39090649
Iteration 214, loss = 0.39049387
Iteration 215, loss = 0.39018263
Iteration 216, loss = 0.38980120
Iteration 217, loss = 0.38944159
Iteration 218, loss = 0.38939923
Iteration 219, loss = 0.38905559
Iteration 220, loss = 0.38874553
Iteration 221, loss = 0.38842909
Iteration 222, loss = 0.38873110
Iteration 223, loss = 0.38716786
Iteration 224, loss = 0.38660240
Iteration 225, loss = 0.38574797
Iteration 226, loss = 0.38530488
Iteration 227, loss = 0.38510250
Iteration 228, loss = 0.38470963
Iteration 229, loss = 0.38446759
Iteration 230, loss = 0.38362930
Iteration 231, loss = 0.38315760
Iteration 232, loss = 0.38281824
Iteration 233, loss = 0.38262545
Iteration 234, loss = 0.38209962
Iteration 235, loss = 0.38277768
Iteration 236, loss = 0.38207322
Iteration 237, loss = 0.38183363
Iteration 238, loss = 0.38099626
Iteration 239, loss = 0.38021690
Iteration 240, loss = 0.38082037
Iteration 241, loss = 0.38069866
Iteration 242, loss = 0.38079773
Iteration 243, loss = 0.37933511
Iteration 244, loss = 0.37831583
Iteration 245, loss = 0.37740728
Iteration 246, loss = 0.37674443
Iteration 247, loss = 0.37625768
Iteration 248, loss = 0.37564597
Iteration 249, loss = 0.37541664
Iteration 250, loss = 0.37503041
Iteration 251, loss = 0.37467099
Iteration 252, loss = 0.37402593
Iteration 253, loss = 0.37357456
Iteration 254, loss = 0.37326729
Iteration 255, loss = 0.37264928
Iteration 256, loss = 0.37235594
Iteration 257, loss = 0.37201137
Iteration 258, loss = 0.37173955
Iteration 259, loss = 0.37127552
Iteration 260, loss = 0.37075150
Iteration 261, loss = 0.37041216
Iteration 262, loss = 0.36996277
Iteration 263, loss = 0.36958847
Iteration 264, loss = 0.36925832
Iteration 265, loss = 0.36889321
Iteration 266, loss = 0.36859103
Iteration 267, loss = 0.36869859
Iteration 268, loss = 0.36813935
Iteration 269, loss = 0.36821140
Iteration 270, loss = 0.36800186
Iteration 271, loss = 0.36756004
Iteration 272, loss = 0.36692060
Iteration 273, loss = 0.36653583
Iteration 274, loss = 0.36614091
Iteration 275, loss = 0.36567036
Iteration 276, loss = 0.36605606
Iteration 277, loss = 0.36522219
Iteration 278, loss = 0.36428627
Iteration 279, loss = 0.36423409
Iteration 280, loss = 0.36380587
Iteration 281, loss = 0.36292353
Iteration 282, loss = 0.36271780
Iteration 283, loss = 0.36237447
Iteration 284, loss = 0.36205187
Iteration 285, loss = 0.36155647
Iteration 286, loss = 0.36147925
Iteration 287, loss = 0.36065627
Iteration 288, loss = 0.36010771
Iteration 289, loss = 0.35983388
Iteration 290, loss = 0.35960898
Iteration 291, loss = 0.35911183
Iteration 292, loss = 0.35886690
Iteration 293, loss = 0.35856768
Iteration 294, loss = 0.35815435
Iteration 295, loss = 0.35770649
Iteration 296, loss = 0.35749904
Iteration 297, loss = 0.35726706
Iteration 298, loss = 0.35706161
Iteration 299, loss = 0.35637585
Iteration 300, loss = 0.35602906
Iteration 301, loss = 0.35586388
Iteration 302, loss = 0.35507476
Iteration 303, loss = 0.35485305
Iteration 304, loss = 0.35459439
Iteration 305, loss = 0.35419497
Iteration 306, loss = 0.35387070
Iteration 307, loss = 0.35361842
Iteration 308, loss = 0.35316468
Iteration 309, loss = 0.35269917
Iteration 310, loss = 0.35240303
Iteration 311, loss = 0.35202973
Iteration 312, loss = 0.35169012
Iteration 313, loss = 0.35137404
Iteration 314, loss = 0.35116155
Iteration 315, loss = 0.35083841
Iteration 316, loss = 0.35060722
Iteration 317, loss = 0.35032341
Iteration 318, loss = 0.35005025
Iteration 319, loss = 0.34970626
Iteration 320, loss = 0.34938870
Iteration 321, loss = 0.34920638
Iteration 322, loss = 0.34872663
Iteration 323, loss = 0.34849002
Iteration 324, loss = 0.34821313
Iteration 325, loss = 0.34797465
Iteration 326, loss = 0.34790102
Iteration 327, loss = 0.34766917
Iteration 328, loss = 0.34774251
Iteration 329, loss = 0.34703804
Iteration 330, loss = 0.34675430
Iteration 331, loss = 0.34625031
Iteration 332, loss = 0.34621711
Iteration 333, loss = 0.34530152
Iteration 334, loss = 0.34508321
Iteration 335, loss = 0.34478938
Iteration 336, loss = 0.34444482
Iteration 337, loss = 0.34428575
Iteration 338, loss = 0.34405920
Iteration 339, loss = 0.34393159
Iteration 340, loss = 0.34364543
Iteration 341, loss = 0.34307701
Iteration 342, loss = 0.34287630
Iteration 343, loss = 0.34255401
Iteration 344, loss = 0.34199431
Iteration 345, loss = 0.34160649
Iteration 346, loss = 0.34184279
Iteration 347, loss = 0.34123034
Iteration 348, loss = 0.34123608
Iteration 349, loss = 0.34135624
Iteration 350, loss = 0.34070434
Iteration 351, loss = 0.34020609
Iteration 352, loss = 0.33919808
Iteration 353, loss = 0.33867216
Iteration 354, loss = 0.33845936
Iteration 355, loss = 0.33834465
Iteration 356, loss = 0.33781724
Iteration 357, loss = 0.33752216
Iteration 358, loss = 0.33703173
Iteration 359, loss = 0.33700486
Iteration 360, loss = 0.33641821
Iteration 361, loss = 0.33604186
Iteration 362, loss = 0.33563316
Iteration 363, loss = 0.33543280
Iteration 364, loss = 0.33518803
Iteration 365, loss = 0.33468182
Iteration 366, loss = 0.33441455
Iteration 367, loss = 0.33411203
Iteration 368, loss = 0.33383064
Iteration 369, loss = 0.33336821
Iteration 370, loss = 0.33316959
Iteration 371, loss = 0.33295257
Iteration 372, loss = 0.33262361
Iteration 373, loss = 0.33248674
Iteration 374, loss = 0.33223668
Iteration 375, loss = 0.33155737
Iteration 376, loss = 0.33132168
Iteration 377, loss = 0.33077105
Iteration 378, loss = 0.33061998
Iteration 379, loss = 0.33031351
Iteration 380, loss = 0.32993613
Iteration 381, loss = 0.32943678
Iteration 382, loss = 0.32930206
Iteration 383, loss = 0.32910417
Iteration 384, loss = 0.32886232
Iteration 385, loss = 0.32837066
Iteration 386, loss = 0.32813752
Iteration 387, loss = 0.32784375
Iteration 388, loss = 0.32753780
Iteration 389, loss = 0.32725941
Iteration 390, loss = 0.32704123
Iteration 391, loss = 0.32710732
Iteration 392, loss = 0.32680916
Iteration 393, loss = 0.32672125
Iteration 394, loss = 0.32637942
Iteration 395, loss = 0.32601403
Iteration 396, loss = 0.32579333
Iteration 397, loss = 0.32544066
Iteration 398, loss = 0.32523358
Iteration 399, loss = 0.32510146
Iteration 400, loss = 0.32481184
Iteration 401, loss = 0.32455038
Iteration 402, loss = 0.32439811
Iteration 403, loss = 0.32399906
Iteration 404, loss = 0.32396706
Iteration 405, loss = 0.32385499
Iteration 406, loss = 0.32339285
Iteration 407, loss = 0.32298347
Iteration 408, loss = 0.32266128
Iteration 409, loss = 0.32263312
Iteration 410, loss = 0.32239123
Iteration 411, loss = 0.32202317
Iteration 412, loss = 0.32186285
Iteration 413, loss = 0.32153570
Iteration 414, loss = 0.32130262
Iteration 415, loss = 0.32095018
Iteration 416, loss = 0.32064305
Iteration 417, loss = 0.32032039
Iteration 418, loss = 0.31972135
Iteration 419, loss = 0.31950537
Iteration 420, loss = 0.31930101
Iteration 421, loss = 0.31905075
Iteration 422, loss = 0.31897691
Iteration 423, loss = 0.31829991
Iteration 424, loss = 0.31796422
Iteration 425, loss = 0.31776245
Iteration 426, loss = 0.31757302
Iteration 427, loss = 0.31725807
Iteration 428, loss = 0.31712285
Iteration 429, loss = 0.31705565
Iteration 430, loss = 0.31712651
Iteration 431, loss = 0.31696002
Iteration 432, loss = 0.31628960
Iteration 433, loss = 0.31517598
Iteration 434, loss = 0.31494847
Iteration 435, loss = 0.31477561
Iteration 436, loss = 0.31466945
Iteration 437, loss = 0.31461262
Iteration 438, loss = 0.31417029
Iteration 439, loss = 0.31365903
Iteration 440, loss = 0.31339926
Iteration 441, loss = 0.31323477
Iteration 442, loss = 0.31300534
Iteration 443, loss = 0.31276823
Iteration 444, loss = 0.31260941
Iteration 445, loss = 0.31236988
Iteration 446, loss = 0.31231457
Iteration 447, loss = 0.31236713
Iteration 448, loss = 0.31208284
Iteration 449, loss = 0.31158747
Iteration 450, loss = 0.31153917
Iteration 451, loss = 0.31145139
Iteration 452, loss = 0.31120284
Iteration 453, loss = 0.31117370
Iteration 454, loss = 0.31128349
Iteration 455, loss = 0.31095281
Iteration 456, loss = 0.31077210
Iteration 457, loss = 0.31072726
Iteration 458, loss = 0.31042405
Iteration 459, loss = 0.30967684
Iteration 460, loss = 0.30976516
Iteration 461, loss = 0.30994588
Iteration 462, loss = 0.30995068
Iteration 463, loss = 0.30940157
Iteration 464, loss = 0.30932655
Iteration 465, loss = 0.30937268
Iteration 466, loss = 0.30878729
Iteration 467, loss = 0.30913983
Iteration 468, loss = 0.30891911
Iteration 469, loss = 0.30847956
Iteration 470, loss = 0.30852237
Iteration 471, loss = 0.30869925
Iteration 472, loss = 0.30811552
Iteration 473, loss = 0.30656715
Iteration 474, loss = 0.30599640
Iteration 475, loss = 0.30546487
Iteration 476, loss = 0.30520805
Iteration 477, loss = 0.30511305
Iteration 478, loss = 0.30519094
Iteration 479, loss = 0.30504465
Iteration 480, loss = 0.30464119
Iteration 481, loss = 0.30465611
Iteration 482, loss = 0.30461776
Iteration 483, loss = 0.30479453
Iteration 484, loss = 0.30386289
Iteration 485, loss = 0.30366835
Iteration 486, loss = 0.30360642
Iteration 487, loss = 0.30295364
Iteration 488, loss = 0.30287182
Iteration 489, loss = 0.30279162
Iteration 490, loss = 0.30225236
Iteration 491, loss = 0.30174710
Iteration 492, loss = 0.30156893
Iteration 493, loss = 0.30122672
Iteration 494, loss = 0.30100013
Iteration 495, loss = 0.30100290
Iteration 496, loss = 0.30070554
Iteration 497, loss = 0.30062299
Iteration 498, loss = 0.30082340
Iteration 499, loss = 0.30065013
Iteration 500, loss = 0.30081044
Iteration 501, loss = 0.30066930
Iteration 502, loss = 0.30110344
Iteration 503, loss = 0.30085049
Iteration 504, loss = 0.30108292
Iteration 505, loss = 0.30057806
Iteration 506, loss = 0.30052356
Iteration 507, loss = 0.30064714'''

log2='''Iteration 1, loss = 0.83361266
Iteration 2, loss = 0.67547816
Iteration 3, loss = 0.54405869
Iteration 4, loss = 0.48204619
Iteration 5, loss = 0.46808518
Iteration 6, loss = 0.46900631
Iteration 7, loss = 0.46809735
Iteration 8, loss = 0.45767553
Iteration 9, loss = 0.44756842
Iteration 10, loss = 0.44924045
Iteration 11, loss = 0.45222028
Iteration 12, loss = 0.44341947
Iteration 13, loss = 0.43158826
Iteration 14, loss = 0.42379594
Iteration 15, loss = 0.41980107
Iteration 16, loss = 0.42129525
Iteration 17, loss = 0.42170511
Iteration 18, loss = 0.41797123
Iteration 19, loss = 0.40397731
Iteration 20, loss = 0.40058090
Iteration 21, loss = 0.40364254
Iteration 22, loss = 0.39708936
Iteration 23, loss = 0.38857594
Iteration 24, loss = 0.38419603
Iteration 25, loss = 0.38204810
Iteration 26, loss = 0.37852015
Iteration 27, loss = 0.36967331
Iteration 28, loss = 0.36295758
Iteration 29, loss = 0.36158721
Iteration 30, loss = 0.35894908
Iteration 31, loss = 0.35318159
Iteration 32, loss = 0.34935229
Iteration 33, loss = 0.34578146
Iteration 34, loss = 0.34171697
Iteration 35, loss = 0.34012814
Iteration 36, loss = 0.33695346
Iteration 37, loss = 0.33363024
Iteration 38, loss = 0.33029933
Iteration 39, loss = 0.32891392
Iteration 40, loss = 0.32846194
Iteration 41, loss = 0.32849098
Iteration 42, loss = 0.32801061
Iteration 43, loss = 0.32290086
Iteration 44, loss = 0.31581770
Iteration 45, loss = 0.31242432
Iteration 46, loss = 0.30926331
Iteration 47, loss = 0.30703132
Iteration 48, loss = 0.30617156
Iteration 49, loss = 0.30237252
Iteration 50, loss = 0.29924910
Iteration 51, loss = 0.29634314
Iteration 52, loss = 0.29601964
Iteration 53, loss = 0.29390896
Iteration 54, loss = 0.29165508
Iteration 55, loss = 0.29207976
Iteration 56, loss = 0.28999894
Iteration 57, loss = 0.28382385
Iteration 58, loss = 0.27827742
Iteration 59, loss = 0.27644167
Iteration 60, loss = 0.27741637
Iteration 61, loss = 0.27780224
Iteration 62, loss = 0.27467300
Iteration 63, loss = 0.27069084
Iteration 64, loss = 0.26612364
Iteration 65, loss = 0.26567230
Iteration 66, loss = 0.26366350
Iteration 67, loss = 0.26124577
Iteration 68, loss = 0.25876606
Iteration 69, loss = 0.25658334
Iteration 70, loss = 0.25514330
Iteration 71, loss = 0.25438659
Iteration 72, loss = 0.25433692
Iteration 73, loss = 0.25137124
Iteration 74, loss = 0.24773203
Iteration 75, loss = 0.24543684
Iteration 76, loss = 0.24383097
Iteration 77, loss = 0.24374861
Iteration 78, loss = 0.24221174
Iteration 79, loss = 0.24154009
Iteration 80, loss = 0.24239077
Iteration 81, loss = 0.24351184
Iteration 82, loss = 0.24089965
Iteration 83, loss = 0.23774743
Iteration 84, loss = 0.23376022
Iteration 85, loss = 0.23096331
Iteration 86, loss = 0.22970088
Iteration 87, loss = 0.22838353
Iteration 88, loss = 0.22804384
Iteration 89, loss = 0.22816214
Iteration 90, loss = 0.22500908
Iteration 91, loss = 0.22251672
Iteration 92, loss = 0.22245335
Iteration 93, loss = 0.22043668
Iteration 94, loss = 0.21884149
Iteration 95, loss = 0.21723948
Iteration 96, loss = 0.21650625
Iteration 97, loss = 0.21502198
Iteration 98, loss = 0.21331036
Iteration 99, loss = 0.21251740
Iteration 100, loss = 0.21217118
Iteration 101, loss = 0.21291413
Iteration 102, loss = 0.21334626
Iteration 103, loss = 0.20942474
Iteration 104, loss = 0.20775095
Iteration 105, loss = 0.20777023
Iteration 106, loss = 0.20673210
Iteration 107, loss = 0.20511935
Iteration 108, loss = 0.20468427
Iteration 109, loss = 0.20519327
Iteration 110, loss = 0.20409307
Iteration 111, loss = 0.20196046
Iteration 112, loss = 0.20112659
Iteration 113, loss = 0.20051858
Iteration 114, loss = 0.19984990
Iteration 115, loss = 0.19913923
Iteration 116, loss = 0.19868901
Iteration 117, loss = 0.19841187
Iteration 118, loss = 0.19927494
Iteration 119, loss = 0.20037103
Iteration 120, loss = 0.20083753
Iteration 121, loss = 0.19793070
Iteration 122, loss = 0.19628474
Iteration 123, loss = 0.19417145
Iteration 124, loss = 0.19448513
Iteration 125, loss = 0.19418880
Iteration 126, loss = 0.19264302
Iteration 127, loss = 0.19205629
Iteration 128, loss = 0.19125849
Iteration 129, loss = 0.19038753
Iteration 130, loss = 0.19109468
Iteration 131, loss = 0.19247500
Iteration 132, loss = 0.19187510
Iteration 133, loss = 0.18975438
Iteration 134, loss = 0.18730394
Iteration 135, loss = 0.18623270
Iteration 136, loss = 0.18774426
Iteration 137, loss = 0.18884161
Iteration 138, loss = 0.18748727
Iteration 139, loss = 0.18413499
Iteration 140, loss = 0.18232753
Iteration 141, loss = 0.18117143
Iteration 142, loss = 0.18106296
Iteration 143, loss = 0.18185803
Iteration 144, loss = 0.18254868
Iteration 145, loss = 0.18366604
Iteration 146, loss = 0.18387785
Iteration 147, loss = 0.18277943
Iteration 148, loss = 0.18083976
Iteration 149, loss = 0.17942310
Iteration 150, loss = 0.17775716
Iteration 151, loss = 0.17672403
Iteration 152, loss = 0.17517383
Iteration 153, loss = 0.17671704
Iteration 154, loss = 0.17867094
Iteration 155, loss = 0.17869999
Iteration 156, loss = 0.17616465
Iteration 157, loss = 0.17424379
Iteration 158, loss = 0.17241759
Iteration 159, loss = 0.17275443
Iteration 160, loss = 0.17393144
Iteration 161, loss = 0.17369373
Iteration 162, loss = 0.17278451
Iteration 163, loss = 0.17134123
Iteration 164, loss = 0.17039708
Iteration 165, loss = 0.16979843
Iteration 166, loss = 0.16957826
Iteration 167, loss = 0.16924353
Iteration 168, loss = 0.16887892
Iteration 169, loss = 0.16853326
Iteration 170, loss = 0.16784207
Iteration 171, loss = 0.16726652
Iteration 172, loss = 0.16698832
Iteration 173, loss = 0.16679002
Iteration 174, loss = 0.16619946
Iteration 175, loss = 0.16587202
Iteration 176, loss = 0.16566096
Iteration 177, loss = 0.16518941
Iteration 178, loss = 0.16441762
Iteration 179, loss = 0.16378862
Iteration 180, loss = 0.16347186
Iteration 181, loss = 0.16427139
Iteration 182, loss = 0.16731347
Iteration 183, loss = 0.16525129
Iteration 184, loss = 0.16135498
Iteration 185, loss = 0.16158460
Iteration 186, loss = 0.16364150
Iteration 187, loss = 0.16251875
Iteration 188, loss = 0.16040575
Iteration 189, loss = 0.15926824
Iteration 190, loss = 0.15904274
Iteration 191, loss = 0.15880301
Iteration 192, loss = 0.15873160
Iteration 193, loss = 0.15924259
Iteration 194, loss = 0.15996187
Iteration 195, loss = 0.16076357
Iteration 196, loss = 0.16002072
Iteration 197, loss = 0.15824603
Iteration 198, loss = 0.15698123
Iteration 199, loss = 0.15654286
Iteration 200, loss = 0.15637991
Iteration 201, loss = 0.15648206
Iteration 202, loss = 0.15651645
Iteration 203, loss = 0.15620062
Iteration 204, loss = 0.15584948
Iteration 205, loss = 0.15588181
Iteration 206, loss = 0.15673362
Iteration 207, loss = 0.15571864
Iteration 208, loss = 0.15378010
Iteration 209, loss = 0.15312761
Iteration 210, loss = 0.15344391
Iteration 211, loss = 0.15454284
Iteration 212, loss = 0.15494009
Iteration 213, loss = 0.15340405
Iteration 214, loss = 0.15203822
Iteration 215, loss = 0.15149040
Iteration 216, loss = 0.15112452
Iteration 217, loss = 0.15096896
Iteration 218, loss = 0.15104985
Iteration 219, loss = 0.15126418
Iteration 220, loss = 0.15162373
Iteration 221, loss = 0.15247747
Iteration 222, loss = 0.15328383
Iteration 223, loss = 0.15264609
Iteration 224, loss = 0.15046793
Iteration 225, loss = 0.14899802
Iteration 226, loss = 0.14980732
Iteration 227, loss = 0.15115581
Iteration 228, loss = 0.15144903
Iteration 229, loss = 0.15068256
Iteration 230, loss = 0.14907862
Iteration 231, loss = 0.14777458
Iteration 232, loss = 0.14942119
Iteration 233, loss = 0.14830377
Iteration 234, loss = 0.14767327
Iteration 235, loss = 0.14711817
Iteration 236, loss = 0.14815591
Iteration 237, loss = 0.14861764
Iteration 238, loss = 0.14781171
Iteration 239, loss = 0.14735037
Iteration 240, loss = 0.14655543
Iteration 241, loss = 0.14570382
Iteration 242, loss = 0.14580827
Iteration 243, loss = 0.14587188
Iteration 244, loss = 0.14587230
Iteration 245, loss = 0.14661329
Iteration 246, loss = 0.14754142
Iteration 247, loss = 0.14618030
Iteration 248, loss = 0.14456036
Iteration 249, loss = 0.14439500
Iteration 250, loss = 0.14468304
Iteration 251, loss = 0.14720323
Iteration 252, loss = 0.14729044
Iteration 253, loss = 0.14609069
Iteration 254, loss = 0.14675508
Iteration 255, loss = 0.14749740
Iteration 256, loss = 0.14667025
Iteration 257, loss = 0.14543551
Iteration 258, loss = 0.14417976
Iteration 259, loss = 0.14287858
Iteration 260, loss = 0.14251172
Iteration 261, loss = 0.14196799
Iteration 262, loss = 0.14154461
Iteration 263, loss = 0.14153020
Iteration 264, loss = 0.14142666
Iteration 265, loss = 0.14168503
Iteration 266, loss = 0.14283297
Iteration 267, loss = 0.14444569
Iteration 268, loss = 0.14476357
Iteration 269, loss = 0.14435135
Iteration 270, loss = 0.14118277
Iteration 271, loss = 0.13979872
Iteration 272, loss = 0.14008346
Iteration 273, loss = 0.13975392
Iteration 274, loss = 0.13949629
Iteration 275, loss = 0.13933550
Iteration 276, loss = 0.13950822
Iteration 277, loss = 0.14003282
Iteration 278, loss = 0.13955475
Iteration 279, loss = 0.13934691
Iteration 280, loss = 0.13948283
Iteration 281, loss = 0.13844011
Iteration 282, loss = 0.13903804
Iteration 283, loss = 0.13952159
Iteration 284, loss = 0.14096522
Iteration 285, loss = 0.14141145
Iteration 286, loss = 0.13949445
Iteration 287, loss = 0.13783844
Iteration 288, loss = 0.13759217
Iteration 289, loss = 0.13770911
Iteration 290, loss = 0.13783020
Iteration 291, loss = 0.13757443
Iteration 292, loss = 0.13723757
Iteration 293, loss = 0.13695992
Iteration 294, loss = 0.13733777
Iteration 295, loss = 0.13767618
Iteration 296, loss = 0.13741934
Iteration 297, loss = 0.13676930
Iteration 298, loss = 0.13647389
Iteration 299, loss = 0.13615294
Iteration 300, loss = 0.13617734
Iteration 301, loss = 0.13660183
Iteration 302, loss = 0.13869149
Iteration 303, loss = 0.13842218
Iteration 304, loss = 0.13637627
Iteration 305, loss = 0.13543388
Iteration 306, loss = 0.13516837
Iteration 307, loss = 0.13516719
Iteration 308, loss = 0.13548021
Iteration 309, loss = 0.13516780
Iteration 310, loss = 0.13536140
Iteration 311, loss = 0.13590657
Iteration 312, loss = 0.13597935
Iteration 313, loss = 0.13577261
Iteration 314, loss = 0.13439246
Iteration 315, loss = 0.13395084
Iteration 316, loss = 0.13400696
Iteration 317, loss = 0.13392424
Iteration 318, loss = 0.13440079
Iteration 319, loss = 0.13442025
Iteration 320, loss = 0.13339304
Iteration 321, loss = 0.13322742
Iteration 322, loss = 0.13395554
Iteration 323, loss = 0.13471424
Iteration 324, loss = 0.13371985
Iteration 325, loss = 0.13276833
Iteration 326, loss = 0.13297614
Iteration 327, loss = 0.13340060
Iteration 328, loss = 0.13281698
Iteration 329, loss = 0.13280139
Iteration 330, loss = 0.13267546
Iteration 331, loss = 0.13235608
Iteration 332, loss = 0.13187834
Iteration 333, loss = 0.13160813
Iteration 334, loss = 0.13183491
Iteration 335, loss = 0.13216107
Iteration 336, loss = 0.13225666
Iteration 337, loss = 0.13237299
Iteration 338, loss = 0.13107936
Iteration 339, loss = 0.13044729
Iteration 340, loss = 0.13062478
Iteration 341, loss = 0.13084771
Iteration 342, loss = 0.13141538
Iteration 343, loss = 0.13143892
Iteration 344, loss = 0.13022659
Iteration 345, loss = 0.13151801
Iteration 346, loss = 0.13293374
Iteration 347, loss = 0.13538584
Iteration 348, loss = 0.14049822
Iteration 349, loss = 0.14000345
Iteration 350, loss = 0.13645244
Iteration 351, loss = 0.13164412
Iteration 352, loss = 0.12889636
Iteration 353, loss = 0.12921997
Iteration 354, loss = 0.13012436
Iteration 355, loss = 0.13034084
Iteration 356, loss = 0.13042955
Iteration 357, loss = 0.13019208
Iteration 358, loss = 0.13014346
Iteration 359, loss = 0.13030564
Iteration 360, loss = 0.12925137
Iteration 361, loss = 0.12845341
Iteration 362, loss = 0.12795694
Iteration 363, loss = 0.12789358
Iteration 364, loss = 0.12776469
Iteration 365, loss = 0.12778800
Iteration 366, loss = 0.12776433
Iteration 367, loss = 0.12774973
Iteration 368, loss = 0.12836861
Iteration 369, loss = 0.12767566
Iteration 370, loss = 0.12812842
Iteration 371, loss = 0.12830662
Iteration 372, loss = 0.12866146
Iteration 373, loss = 0.12965075
Iteration 374, loss = 0.13042914
Iteration 375, loss = 0.12953268
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
the mse on the train data :0.25612562025442526, on the test data: 0.4890469976347465'''
loss1=[]
import re
for line in log1.split('\n'):
    loss=re.findall(r'= 0\.\d*',line)
    # print(loss)
    if loss:
        loss=float(loss[0][2:])
        loss1.append(loss)


loss2=[]
import re
for line in log2.split('\n'):
    loss=re.findall(r'= 0\.\d*',line)
    # print(loss)
    if loss:
        loss=float(loss[0][2:])
        loss2.append(loss)


from matplotlib import pyplot as plt
plt.plot(loss1)
plt.plot(loss2)
plt.legend(['wo momentum','0.9 momentum'])
plt.ylabel('train loss')
plt.xlabel('Iteration')
plt.show()