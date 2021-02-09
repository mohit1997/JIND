from sys import argv
from pathlib import Path
import numpy as np
import time as tm
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import torch, sys, os, pdb
import pandas as pd
from torch import optim
from torch.autograd import Variable
from .utils import DataLoaderCustom, ConfusionMatrixPlot, compute_ap, normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .models import Classifier, Discriminator, ClassifierBig
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

class SVMReject:

    def __init__(self, gene_mat, cell_labels, path):
        self.class2num = None
        self.num2class = None
        self.reduced_features = None
        self.reduce_method = None
        self.model = None
        self.preprocessed = False
        self.path = path
        self.logt = False
        self.count_normalize = False
        os.system('mkdir -p {}'.format(self.path))

        self.raw_features = gene_mat.values
        self.cell_ids = list(gene_mat.index)
        self.gene_names = list(gene_mat.columns)
        

        classes = list(set(cell_labels))
        classes.sort()
        self.classes = classes
        self.n_classes = len(classes)

        self.class2num = class2num = {c: i for (i, c) in enumerate(classes)}
        self.class2num['Unassigned'] = self.n_classes

        self.num2class = num2class = {i: c for (i, c) in enumerate(classes)}
        self.num2class[self.n_classes] = 'Unassigned'

        self.labels = np.array([class2num[i] for i in cell_labels])
        self.val_stats = None
        self.scaler = None

    def preprocess(self, count_normalize=False, target_sum=1e4, logt=True):
        self.logt = logt
        self.count_normalize = count_normalize
        if count_normalize:
            print('Normalizing counts ...')
            self.raw_features = self.raw_features / (np.sum(self.raw_features, axis=1, keepdims=True) + 1e-5) * target_sum
        if logt:
            print('Applying log transformation ...')
            self.raw_features = np.log(1 + self.raw_features)

    def get_features(self, gene_mat):
        features = gene_mat.values
        if self.count_normalize:
            features = features / (np.sum(features, axis=1, keepdims=True) + 1e-5) * 1e4
        if self.logt:
            features = np.log(1+features)
        if self.reduce_method is not None:
            if self.reduce_method == "Var":
                selected_genes = [gene_mat.columns[i] for i in self.variances]
                if selected_genes != self.selected_genes:
                    print("Reorder the genes for the target batch in the same order as the source batch")
                    sys.exit()
                features = features[:, self.variances]
            elif self.reduce_method == "PCA":
                features = self.pca.transform(features)
        if self.scaler is not None:
            self.test_scaler = StandardScaler()
            features = self.test_scaler.fit_transform(features)

        return features


    def dim_reduction(self, num_features=5000, method='var', save_as=None):
        dim_size = num_features
        self.reduce_method = method

        if method.lower() == 'pca':
            print('Performing PCA ...')
            self.pca = PCA(n_components=dim_size)
            self.reduced_features = self.pca.fit_transform(self.raw_features)
            if save_as is not None:
                np.save('{}_{}'.format(save_as, method), self.reduced_features)

        elif method.lower() == 'var':
            print('Variance based reduction ...')
            self.variances = np.argsort(-np.var(self.raw_features, axis=0))[:dim_size]
            self.reduced_features = self.raw_features[:, self.variances]
            self.selected_genes = [self.gene_names[i] for i in self.variances]
            if save_as is not None:
                np.save('{}_{}'.format(save_as, method), self.reduced_features)

    def normalize(self):
        scaler = MinMaxScaler((0, 1))
        self.reduced_features = scaler.fit_transform(self.reduced_features)
        self.scaler = scaler


    def train_classifier(self, use_red, config, cmat=True):
        if use_red:
            if self.reduced_features is None:
                print("Please run obj.dim_reduction() or use use_red=False")
                sys.exit()
            features = self.reduced_features
        else:
            features = self.raw_features

        labels = self.labels

        np.random.seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=config['val_frac'], shuffle=True, random_state=config['seed'])

        Classifier = LinearSVC(random_state=config['seed'])
        model = CalibratedClassifierCV(Classifier)
        model.fit(features, labels)

        y_pred = model.predict_proba(X_val)
        y_true = y_val

        self.val_stats = {'pred': y_pred, 'true': y_true}

        if cmat:
            # Plot validation confusion matrix
            self.plot_cfmt(self.val_stats['pred'], self.val_stats['true'], 0.05, 'val_cfmt.pdf')

        # Finally keep the best model
        self.model = model

    def predict(self, test_gene_mat, test=False):
        features = self.get_features(test_gene_mat)

        model = self.model

        y_pred = model.predict_proba(features)

        return y_pred


    def evaluate(self, test_gene_mat, test_labels, frac=0.05, name=None, return_log=False):
        y_pred = self.predict(test_gene_mat)
        y_true = np.array([self.class2num[i] if (i in self.class2num.keys()) else (self.n_classes + 1) for i in test_labels])
        if frac != 0:
            preds = self.filter_pred(y_pred, frac)
        else:
            preds = np.argmax(y_pred, axis=1)
        pretest_acc = (y_true == np.argmax(y_pred, axis=1)).mean() 
        test_acc = (y_true == preds).mean()
        ind = preds != self.n_classes
        pred_acc = (y_true[ind] == preds[ind]).mean()
        filtered = 1 - np.mean(ind)
        print('Test Acc Pre {:.4f} Post {:.4f} Eff {:.4f} Filtered {:.4f}'.format(pretest_acc, test_acc, pred_acc, filtered))

        if name is not None:
            cm = normalize(confusion_matrix(y_true,
                            preds,
                            labels=np.arange(0, max(np.max(y_true)+1, np.max(preds)+1, self.n_classes+1))
                            ),
                            normalize='true')
            cm = np.delete(cm, (self.n_classes), axis=0)
            if cm.shape[1] > (self.n_classes+1):
                cm = np.delete(cm, (self.n_classes+1), axis=1)
            aps = np.zeros((len(cm), 1))
            aps[:self.n_classes] = np.array(compute_ap(y_true, y_pred)).reshape(-1, 1)
            cm = np.concatenate([cm, aps], axis=1)

            class_labels = list(self.class2num.keys()) +['Novel'] + ['AP']
            cm_ob = ConfusionMatrixPlot(cm, class_labels)
            factor = max(1, len(cm) // 10)
            fig = plt.figure(figsize=(10*factor,8*factor))
            cm_ob.plot(values_format='0.2f', ax=fig.gca())

            plt.title('Accuracy Pre {:.3f} Post {:.3f} Eff {:.3f} Filtered {:.3f} mAP {:.3f}'.format(pretest_acc, test_acc, pred_acc, filtered, np.mean(aps[:self.n_classes])), fontsize=16)
            plt.tight_layout()
            plt.savefig('{}/{}'.format(self.path, name))

        predictions = [self.num2class[i] for i in preds]
        raw_predictions = [self.num2class[i] for i in np.argmax(y_pred, axis=1)]
        predicted_label = pd.DataFrame({"cellname": test_gene_mat.index,
                                        "raw_predictions": raw_predictions,
                                        "predictions": predictions,
                                        "labels": test_labels})
        
        predicted_label = predicted_label.set_index("cellname")
        if return_log:
            return predicted_label, 'Test Acc Pre {:.4f} Post {:.4f} Eff {:.4f} Filtered {:.4f}'.format(pretest_acc, test_acc, pred_acc, filtered)
        return predicted_label

    def plot_cfmt(self, y_pred, y_true, frac=0.05, name=None):
        if frac != 0:
            preds = self.filter_pred(y_pred, frac)
        else:
            preds = np.argmax(y_pred, axis=1)
        pretest_acc = (y_true == np.argmax(y_pred, axis=1)).mean() 
        test_acc = (y_true == preds).mean()
        ind = preds != self.n_classes
        pred_acc = (y_true[ind] == preds[ind]).mean()
        filtered = 1 - np.mean(ind)
        print('Test Acc Pre {:.4f} Post {:.4f} Eff {:.4f} Filtered {:.4f}'.format(pretest_acc, test_acc, pred_acc, filtered))

        if name is not None:
            cm = normalize(confusion_matrix(y_true,
                            preds,
                            labels=np.arange(0, max(np.max(y_true)+1, np.max(preds)+1, self.n_classes+1))
                            ),
                            normalize='true')
            cm = np.delete(cm, (self.n_classes), axis=0)
            if cm.shape[1] > (self.n_classes+1):
                cm = np.delete(cm, (self.n_classes+1), axis=1)
            aps = np.zeros((len(cm), 1))
            aps[:self.n_classes] = np.array(compute_ap(y_true, y_pred)).reshape(-1, 1)
            cm = np.concatenate([cm, aps], axis=1)

            class_labels = list(self.class2num.keys()) +['Novel'] + ['AP']
            cm_ob = ConfusionMatrixPlot(cm, class_labels)
            factor = max(1, len(cm) // 10)
            fig = plt.figure(figsize=(10*factor,7*factor))
            cm_ob.plot(values_format='0.2f', ax=fig.gca())

            plt.title('Accuracy Pre {:.3f} Post {:.3f} Eff {:.3f} Filtered {:.3f} mAP {:.3f}'.format(pretest_acc, test_acc, pred_acc, filtered, np.mean(aps[:self.n_classes])), fontsize=16)
            plt.tight_layout()
            plt.savefig('{}/{}'.format(self.path, name))


    def get_thresholds(self, outlier_frac):

        thresholds = 0.7*np.ones((self.n_classes))
        # probs_train = self.val_stats['pred']
        # y_train = self.val_stats['true']
        # for top_klass in range(self.n_classes):
        #     ind = (np.argmax(probs_train, axis=1) == top_klass) #& (y_train == top_klass)

        #     if np.sum(ind) != 0:
        #         best_prob = np.max(probs_train[ind], axis=1)
        #         best_prob = np.sort(best_prob)
        #         l = int(outlier_frac * len(best_prob)) + 1

        #         thresholds[top_klass] = best_prob[l]

        return thresholds


    def filter_pred(self, pred, outlier_frac):
        thresholds = self.get_thresholds(outlier_frac)

        pred_class = np.argmax(pred, axis=1)
        prob_max = np.max(pred, axis=1)

        ind = prob_max < thresholds[pred_class]
        pred_class[ind] = self.n_classes # assign unassigned class
        return pred_class

    def get_TSNE(self, features):
        pca = PCA(n_components=30)
        reduced_feats = pca.fit_transform(features)
        embeddings = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(reduced_feats)
        return embeddings

    def to_pickle(self, name):
        self.raw_features = None
        self.reduced_features = None
        with open('{}/{}'.format(self.path, name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

def main():
    import pickle

    data = pd.read_pickle('data/data_nocovgene.pkl')
    cell_ids = np.arange(len(data))
    np.random.seed(0)

    # batches = list(set(data['sample']))
    # batches.sort()
    # l = int(0.5*len(batches))
    batches = ['C51', 'C141', 'C145', 'C146']
    test_batches = ['C142', 'C144', 'C52', 'C100', 'C143', 'C148']
    train_data = data[data['sample'].isin(batches)].copy()
    test_data = data[data['sample'].isin(test_batches)].copy()

    # train_labels = train_data['labels']
    # # train_gene_mat =  train_data.drop(['labels', 'batch'], 1)

    # test_labels = test_data['labels']
    # # test_gene_mat =  test_data.drop(['labels', 'batch'], 1)

    # common_labels = list(set(train_labels) & set(test_labels))
    # common_labels.sort()

    # train_data = train_data[train_data['labels'].isin(common_labels[:-3])].copy()
    # test_data = data[data['batch'].isin(batches[1:2])].copy()
    # test_data = test_data[test_data['labels'].isin(common_labels)].copy()
    # # test_data = test_data[test_data['labels'].isin(common_labels)].copy()

    # train_labels = train_data['labels']
    # train_gene_mat =  train_data.drop(['labels', 'batch'], 1)

    # test_labels = test_data['labels']
    # test_gene_mat =  test_data.drop(['labels', 'batch'], 1)

    # # assert (set(train_labels)) == (set(test_labels))
    # common_labels.sort()
    # testing_set = list(set(test_labels))
    # testing_set.sort()
    # print("Selected Common Labels", common_labels)
    # print("Test Labels", testing_set)

    # cell_ids = np.arange(len(data))
    # np.random.seed(0)
    # np.random.shuffle(cell_ids)
    # l = int(0.7*len(cell_ids))

    # train_data = data.iloc[cell_ids[:l]]
    train_labels = train_data['celltype']
    train_gene_mat =  train_data.drop(["sample", "sample_new", "group", "disease", "hasnCoV", "cluster", "celltype"], 1)

    # test_data = data.iloc[cell_ids[l:]]
    test_labels = test_data['celltype']
    test_gene_mat =  test_data.drop(["sample", "sample_new", "group", "disease", "hasnCoV", "cluster", "celltype"], 1)


    obj = scRNALibSVM(train_gene_mat, train_labels, path="covid_results")
    obj.preprocess()
    obj.dim_reduction(5000, 'Var')

    train_config = {'val_frac': 0.2, 'seed': 0}
    
    obj.train_classifier(True, train_config)

    obj.raw_features = None
    obj.reduced_features = None
    with open('covid_results/scRNALibSVM_obj.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    obj.evaluate(test_gene_mat, test_labels, frac=0.05, name="testcfmtSVM.pdf")


if __name__ == "__main__":
    start = tm.time()
    main()
    print("Time Taken {}".format(tm.time() - start))
