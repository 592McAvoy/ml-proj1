import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
import sklearn.metrics as metrics
import os
from joblib import dump, load
import logging
import threading
from scipy.special import softmax
import scipy.io
# logging.getLogger('matplotlib.font_manager').disabled = True
# logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
#                     filename='saved/log/SVM/svm.log',
#                     filemode='w',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#                     format='%(asctime)s [line:%(lineno)d] - : %(message)s'
#                     )
def load_data(mode='train', N=7000, target_cls=1):
    mat = scipy.io.loadmat('data/SVHN/{}_32x32.mat'.format(mode))
    X = mat['X'].transpose(3, 0, 1, 2)
    y = mat['y'][:,0]

    
    print(X.shape)

    pos_cls = target_cls
    neg_cls = target_cls%10+1
    print('Positive Number:{}\tNegative Number:{}'.format(pos_cls, neg_cls))
    new_X = []
    new_y = []
    for lab in [pos_cls, neg_cls]:
        new_X.append(X[y==lab][:N//2])
        new_y.append(y[y==lab][:N//2])
        print(new_X[-1].shape)
    new_X = np.concatenate(new_X)
    new_y = np.concatenate(new_y) 
    new_X = new_X.reshape((N,-1))
    print(new_X.shape)  
  

    return new_X, new_y
    


def main():
    n_train = 10000
    n_test = 5000
    X_train, y_train = load_data(mode='train', N=n_train)
    X_test, y_test = load_data(mode='test', N=n_test)

    print(X_train.shape, y_train.shape)

    print('Start Fitting')
    # clf = SVC(kernel='rbf')
    clf = KernelRidge(alpha=1.0, kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    lab = np.zeros_like(y_pred)
    lab[y_pred<0.5] = 2
    lab[y_pred>=0.5] = 1
    print('End Fitting')

    acc = metrics.accuracy_score(y_test, lab)
    print(acc)

if __name__ == '__main__':
    main()
# def train(loader, true_label=0, save_path=None):
#     X_train, y_train = loader.get_subproblem(true_label)
#     clf_rbf = SVC(kernel='rbf')
#     print('[Start] Postive label: {}'.format(true_label))
#     clf_rbf.fit(X_train, y_train)
#     dump(clf_rbf, save_path)
#     print('[Done] Postive label: {}'.format(true_label))


# def paralell_train():
#     # data
#     train_loader = DataLoader(mode='train')
#     os.makedirs(ckpt_path, exist_ok=True)

#     # one-vs-rest
#     threads = []
#     for x in range(3):
#         t = threading.Thread(target=train, args=(
#             train_loader, x-1, os.path.join(ckpt_path, '{}.joblib'.format(x))))
#         threads.append(t)

#     for thr in threads:
#         thr.start()

#     for thr in threads:
#         if thr.isAlive():
#             thr.join()


# def test(X_test):
#     n_test = X_test.shape[0]

#     svm0 = load(os.path.join(ckpt_path, '0.joblib'))
#     svm1 = load(os.path.join(ckpt_path, '1.joblib'))
#     svm2 = load(os.path.join(ckpt_path, '2.joblib'))

#     score0 = svm0.decision_function(X_test).reshape(n_test, 1)
#     score1 = svm1.decision_function(X_test).reshape(n_test, 1)
#     score2 = svm2.decision_function(X_test).reshape(n_test, 1)

#     tmp_score = np.concatenate([score0, score1, score2], axis=1)
#     y_pred = np.array(np.argmax(tmp_score, axis=1) - 1)

#     return tmp_score, y_pred


# def metric(y_true, y_pred, y_score):
#     logging.info('Accuracy: {:.6f}'.format(
#         metrics.accuracy_score(y_true, y_pred)))
#     # print('F1 score: {:.6f}')
#     target_names = ['class -1', 'class 0', 'class 1']
#     logging.info(metrics.classification_report(
#         y_true, y_pred, target_names=target_names))

#     skplt.metrics.plot_roc(y_true, y_score, figsize=(8,6))
#     plt.show()
#     plt.savefig('prob1_roc.png')


# if __name__ == '__main__':
#     ckpt_path = 'ckpts/prob1'

#     # train
#     if not os.path.exists(os.path.join(ckpt_path, '0.joblib')):
#         paralell_train()

#     # test
#     test_loader = DataLoader(mode='test')
#     X_test, y_test = test_loader.get_data()
#     if not os.path.exists(os.path.join(ckpt_path, 'y_pred.npy')):
#         tmp_score, y_pred = test(X_test)
#         y_score = softmax(tmp_score, axis=1)
#         np.save(os.path.join(ckpt_path, 'y_pred.npy'), y_pred)
#         np.save(os.path.join(ckpt_path, 'y_score.npy'), y_score)
#     else:
#         y_pred = np.load(os.path.join(ckpt_path, 'y_pred.npy'))
#         y_score = np.load(os.path.join(ckpt_path, 'y_score.npy'))
  

#     metric(y_test, y_pred, y_score)
