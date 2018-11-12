#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import glob
import math
import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('th')


#クラス数
N_CLASS = 18

#seed値
np.random.seed(2018)

#使用する画像サイズ
img_rows, img_cols = 224, 224

#画像データ 1枚の読み込み鳥サイズを行う
def get_im(path):
    
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_cols, img_rows))

    return resized

#データの読み込み、正規化、シャッフルを行う
def read_train_data(ho=0, kind='train'):

    train_data = []
    train_target = []

    #学習データ読み込み
    for j in range(0,N_CLASS):

        path = '../../../data/'
        path += '%s/%i/*/%i/*.JPG'%(kind, ho, j)

        files = sorted(glob.glob(path))

        for fl in files:
            #ファル名取得
            flbase = os.path.basename(fl)

            # 画像 1枚 読み込み
            img = get_im(fl)
            img = np.array(img, dtype=np.float32)

            # 正規化(GCN)実行
            img -= np.mean(img)
            img /= np.std(img)

            train_data.append(img)
            train_target.append(j)

    # 読み込んだデータを numpy の array に変換
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.uint8)

    # (レコード数,縦,横,channel数) を (レコード数,channel数,縦,横) に変換
    train_data = train_data.transpose((0, 3, 1, 2))

    # target を 6次元のデータに変換。
    # ex) 1 -> 0,1,0,0,0,0   2 -> 0,0,1,0,0,0
    train_target = np_utils.to_categorical(train_target, 18)

    # データをシャッフル
    perm = np.random.permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]

    return train_data, train_target

#テストデータ読み込み
def load_test(test_class, aug_i):
    path = '../../../data/test/%i/%i/*.JPG'%(aug_i, test_class)

    files = sorted(glob.glob(path))
    X_test = []
    X_test_id = []

    for fl in files:
        flbase = os.path.basename(fl)

        img = get_im(fl)
        img = np.array(img, dtype=np.float32)

        # 正規化(GCN)実行
        img -= np.mean(img)
        img /= np.std(img)

        X_test.append(img)
        X_test_id.append(flbase)

    # 読み込んだデータを numpy の array に変換
#    print(len(X_test))
    test_data = np.array(X_test, dtype=np.float32)

    # (レコード数,縦,横,channel数) を (レコード数,channel数,縦,横) に変換
    test_data = test_data.transpose((0, 3, 1, 2))

    return test_data, X_test_id

# VGG-16 モデル 作成
def vgg16_model():

    input_tensor = Input(shape=(3, 224, 224))
    vgg16_model_fine = VGG16(include_top=True, weights='imagenet', input_tensor=input_tensor, input_shape=None)

    #16層の最後の層（fc 1000）を削除
    vgg16_model_fine.layers.pop()
    # fine tune用に、各レイヤーの重みを固定にしないための処理
    for layer in vgg16_model_fine.layers:
        layer.trainable = True

    #新たに最後の層用に6クラス分類用の出力層を作成
    last = vgg16_model_fine.layers[-1].output
    x = Dense(18, activation='softmax')(last)

    # モデルを統合　
    model = Model(inputs=vgg16_model_fine.input, outputs=x)

    #model.summary()

    # ロス計算や勾配計算に使用する式を定義する。
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    return model


#モデルの構成と重みを読み込む
def read_model(ho, modelStr='', epoch='00'):
    #モデルの構成のファイル名
    json_name = 'architecture_%s_%i.json'%(modelStr, ho)
    #モデル重みのファイル名
    weight_name = 'model_weights_%s_%i_%s.h5'%(modelStr, ho, epoch)

    # モデルの構成を読込み、jsonからモデルオブジェクトへ変換
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    # モデルオブジェクトへ重みを読み込む
    model.load_weights(os.path.join('cache', weight_name))

    return model


# モデルの構成を保存
def save_model(model, ho, modelStr=''):
    # モデルオブジェクトをjson形式に変換
    json_string = model.to_json()
    # カレントディレクトリにcacheディレクトリがなければ作成
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    # モデルの構成を保存するためのファイル名
    json_name = 'architecture_%s_%i.json'%(modelStr, ho)
    # モデル構成を保存
    open(os.path.join('cache', json_name), 'w').write(json_string)


def run_train(modelStr=''):
    # Cacheディレクトリの作成
    if not os.path.isdir('./cache'):
        os.mkdir('./cache')

    # HoldOut 2回行う
    for ho in range(2):

        # モデルの作成
        model = vgg16_model()

        # trainデータ読み込み
        t_data, t_target = read_train_data(ho, 'train')
        v_data, v_target = read_train_data(ho, 'valid')

        print(t_data.shape)

        # CheckPointを設定。エポック毎にweightsを保存する。
        cp = ModelCheckpoint('./cache/model_weights_%s_%i_{epoch:02d}.h5'%(modelStr, ho),
        monitor='val_loss', save_best_only=False)

        # train実行
        model.fit(t_data, t_target, batch_size=32,
                  epochs=20,
                  verbose=1,
                  validation_data=(v_data, v_target),
                  shuffle=True,
                  callbacks=[cp])


        # モデルの構成を保存
        save_model(model, ho, modelStr)


# テストデータのクラスを推測
def run_test(modelStr, epoch1, epoch2):
    """
    # クラス名取得
    columns = []
    for line in open("../../data/Caltech-101/label.csv", 'r'):
        sp = line.split(',')
        for column in sp:
            columns.append(column.split(":")[1])
    """

    #クラス名取得
    columns = []
    for i in range(0,N_CLASS):
        columns.append(i)

    # テストデータが各クラスに分かれているので、
    # 1クラスずつ読み込んで推測を行う。
    for test_class in range(0, N_CLASS):

        yfull_test = []

        # データ拡張した画像を読み込むために5回繰り返す
        for aug_i in range(0,5):

            # テストデータを読み込む
            test_data, test_id = load_test(test_class, aug_i)

            #print test_id

            # HoldOut 2回繰り返す
            for ho in range(2):

                if ho == 0:
                    epoch_n = epoch1
                else:
                    epoch_n = epoch2

                # 学習済みモデルの読み込み
                model = read_model(ho, modelStr, epoch_n)

                # 推測の実行
                test_p = model.predict(test_data, batch_size=128, verbose=1)

                yfull_test.append(test_p)

        # 推測結果の平均化
        test_res = np.array(yfull_test[0])
        for i in range(1,10):
            test_res += np.array(yfull_test[i])
        test_res /= 10

        # 推測結果とクラス名、画像名を合わせる
        result1 = pd.DataFrame(test_res, columns=columns)
        result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)

        # 順番入れ替え
        #result1 = result1.ix[:,[6, 0, 1, 2, 3, 4, 5]]
        print(result1)

        if not os.path.isdir('subm'):
            os.mkdir('subm')
        sub_file = './subm/result_%s_%i.csv'%(modelStr, test_class)

        # 最終推測結果を出力する
        result1.to_csv(sub_file, index=False)

        # 推測の精度を測定する。
        # 一番大きい値が入っているカラムがtest_classであるレコードを探す
        one_column = np.where(np.argmax(test_res, axis=1)==test_class)
        print ("正解数　　" + str(len(one_column[0])))
        print ("不正解数　" + str(test_res.shape[0] - len(one_column[0])))


# 実行した際に呼ばれる
if __name__ == '__main__':

    # 引数を取得
    # [1] = train or test
    # [2] = test時のみ、使用Epoch数 1
    # [3] = test時のみ、使用Epoch数 2
    param = sys.argv

    if len(param) < 2:
        print("Usage: python VGG_16.py [train, test] [1] [2]")
        sys.exit(1)

    # train or test
    run_type = param[1]

    if run_type == 'train':
        run_train('VGG_16')
    elif run_type == 'test':
        # testの場合、使用するエポック数を引数から取得する
        if len(param) == 4:
            epoch1 = "%02d"%(int(param[2])-1)
            epoch2 = "%02d"%(int(param[3])-1)
            run_test('VGG_16', epoch1, epoch2)
        else:
            print("Usage: python VGG_16.py [train, test] [1] [2]")
            sys.exit(1)






