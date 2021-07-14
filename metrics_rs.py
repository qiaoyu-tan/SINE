import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
from collections import OrderedDict
import faiss
from sklearn import metrics
import multiprocessing as mp
import time


def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def pdist2(v1, v2):
    """
    v1: m x h    m: number of example images   h: dimension of hidden layer (e.g. 48)
    v2: 1 x h
    typ: type of distance. supported: 'hamming', 'euclidean'
    """
    # if typ == 'euclidean':
    #     return np.sum(abs(v1-v2)!=0)
    # if typ == 'hamming':
    tt = 1.0 * np.sum(v1 != v2, axis=1) / v2.shape[1]
    # tt = 1.0 * np.sum(v1 != v2, axis=1)

    return tt


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r, cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))

def construct_bit_buckets(item_sparse_code):
    item_len, dim = item_sparse_code.shape

    interest_group = {i: [] for i in range(dim)}

    for i in range(dim):
        sparse_code_col = np.squeeze(item_sparse_code[:, i].toarray())
        group_index = np.where(sparse_code_col == 1)
        group_index = list(group_index[0])

        interest_group[i] = group_index

    return interest_group

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.5
    return res


def predict_matrix(out_test):
    pred_matrix = out_test[0]
    train_label = out_test[1]
    train_label = 1 - train_label
    pred_matrix = np.multiply(pred_matrix, train_label)
    return pred_matrix


def evaluate_one_user(topk, ranking_user, pos_test_items):
    recall_, ndcg_, precision, f1 = [], [], [], []
    r = []
    max_k = max(topk)
    ranking_user = ranking_user[0:max_k]
    for i in range(len(ranking_user)):
        item_ = ranking_user[i]
        if item_ in pos_test_items:
            r.append(1)
        else:
            r.append(0)
    for K in topk:
        recall_tem = recall_at_k(r, K, len(pos_test_items))
        recall_.append(recall_tem)
        ndcg_.append(ndcg_at_k(r, K))
        precision_tem = precision_at_k(r, K)
        precision.append(precision_tem)

        f1.append(F1(precision_tem, recall_tem))

    return recall_, ndcg_, precision, f1


def evaluate_one_user_gmnn(topk, ranking_user, pos_test_items):
    recall_, ndcg_, hit_rate_ = [], [], []
    r = []
    max_k = max(topk)
    ranking_user = ranking_user[0:max_k]
    for i in range(len(ranking_user)):
        item_ = ranking_user[i]
        if item_ in pos_test_items:
            r.append(1)
        else:
            r.append(0)
    for K in topk:
        recall_tem = recall_at_k(r, K, len(pos_test_items))
        hit_rate = hit_at_k(r, K)
        recall_.append(recall_tem)
        ndcg_.append(ndcg_at_k(r, K))
        hit_rate_.append(hit_rate)

    return recall_, ndcg_, hit_rate_


def parse_config(flags, verbose=False):
    config = OrderedDict(sorted(flags.__flags.items()))
    # if flags.python_version == 3:
    for k, v in config.items():
        config[k] = v
    if verbose:
        print(">>>>> params setting: ")
        for k, v in config.items():
            print('{}:'.format(k), config[k])


def evaluate_training_(preds, labels):
    # auc = metrics.roc_auc_score(labels, preds)
    recall = metrics.recall_score(labels, preds >= 0.5)
    precision = metrics.precision_score(labels, preds >= 0.5)
    f1 = metrics.f1_score(labels, preds >= 0.5)
    print(
        # 'eval_auc = {:.5f}'.format(auc),
        'eval_f1 = {:.5f}'.format(f1),
        'eval rec: {:.4f}'.format(recall),
        'eval prec: {:.4f}'.format(precision))


def embedding_generation_batch(input_data, sess, model, dim, sparse=False, placeholders=None, beta=None):
    user_result = None
    item_result = None
    if not sparse:
        user_embedding = sp.dok_matrix((input_data.user_len, dim), dtype=float)
        feed_dict_list, u_batch_list = input_data.user_test_batch()
        for i in range(len(feed_dict_list)):
            feed_dict = feed_dict_list[i]
            user_embedding_ = sess.run(model.user_output, feed_dict=feed_dict)
            u_batch = u_batch_list[i]
            user_embedding[u_batch, :] = user_embedding_

        user_embedding = user_embedding.tocsr()

        item_embedding = sp.dok_matrix((input_data.item_len, dim), dtype=float)
        feed_dict_list, i_batch_list = input_data.item_test_batch()
        for i in range(len(feed_dict_list)):
            feed_dict = feed_dict_list[i]
            item_embedding_ = sess.run(model.item_output, feed_dict=feed_dict)
            i_batch = i_batch_list[i]
            item_embedding[i_batch, :] = item_embedding_

        item_embedding = item_embedding.tocsr()
        user_result = user_embedding
        item_result = item_embedding
    else:
        user_embedding = sp.dok_matrix((input_data.user_len, dim), dtype=float)
        user_embedding_spar = sp.dok_matrix((input_data.user_len, dim), dtype=float)
        feed_dict_list, u_batch_list = input_data.user_test_batch()
        for i in range(len(feed_dict_list)):
            feed_dict = feed_dict_list[i]
            feed_dict.update({placeholders['beta']: beta})
            outs = sess.run([model.user_output_con, model.user_output_spar], feed_dict=feed_dict)
            u_batch = u_batch_list[i]
            user_embedding[u_batch, :] = outs[0]
            user_embedding_spar[u_batch, :] = outs[1]

        user_embedding = user_embedding.tocsr()
        user_embedding_spar = user_embedding_spar.tocsr()

        item_embedding = sp.dok_matrix((input_data.item_len, dim), dtype=float)
        item_embedding_spar = sp.dok_matrix((input_data.item_len, dim), dtype=float)
        feed_dict_list, i_batch_list = input_data.item_test_batch()
        for i in range(len(feed_dict_list)):
            feed_dict = feed_dict_list[i]
            feed_dict.update({placeholders['beta']: beta})
            outs = sess.run([model.item_output_con, model.item_output_spar], feed_dict=feed_dict)
            i_batch = i_batch_list[i]
            item_embedding[i_batch, :] = outs[0]
            item_embedding_spar[i_batch, :] = outs[1]

        item_embedding = item_embedding.tocsr()
        item_embedding_spar = item_embedding_spar.tocsr()
        user_result = [user_embedding, user_embedding_spar]
        item_result = [item_embedding, item_embedding_spar]
    return user_result, item_result


def evaluate_embedding(user_embedding, item_embedding, train_label_dict, test_label_dict, topk):
    precision, recall_, f1, ndcg, auc = [], [], [], [], []
    for user_ in test_label_dict.keys():
        training_items = train_label_dict[user_]
        pos_test_items = test_label_dict[user_]

        embed_user = user_embedding[user_]
        pred_user = embed_user.dot(item_embedding.transpose()).toarray()[0]
        pred_user = sigmoid(pred_user)
        pred_user[training_items] = 0.
        ranking_user = np.argsort(- pred_user)

        recall_tmp, ndcg_tmp, precision_tmp, f1_tmp = evaluate_one_user(topk, ranking_user, pos_test_items)

        recall_.append(recall_tmp)
        ndcg.append(ndcg_tmp)
        precision.append(precision_tmp)
        f1.append(f1_tmp)

        true_labels = np.zeros((len(pred_user, )))
        true_labels[pos_test_items] = 1.

        try:
            auc.append(roc_auc_score(true_labels, pred_user))
        except Exception:
            pass

    recall_ = np.array(recall_)
    ndcg = np.array(ndcg)
    auc = np.array(auc)
    precision = np.array(precision)
    f1 = np.array(f1)

    recall_ = np.mean(recall_, axis=0)
    precision = np.mean(precision, axis=0)
    f1 = np.mean(f1, axis=0)
    ndcg = np.mean(ndcg, axis=0)
    auc = np.mean(auc)

    return recall_, ndcg, auc, precision, f1


def evaluate_embedding_parellel_cont(embed_user, item_embedding, pos_test_items, user_len):
    precision, recall_, f1, ndcg, auc = [], [], [], [], None
    topk = [10, 30, 50, 70, 90, 100, 120, 150, 200]
    # try:
    #     embed_user = np.array(embed_user[0])
    # except:
    #     embed_user = np.array(embed_user)

    embed_user = np.array(embed_user)
    pred_user = embed_user.dot(item_embedding.transpose())
    pred_user = sigmoid(pred_user)
    ranking_user = np.argsort(- pred_user)
    # pos_test_items = [i - user_len for i in pos_test_items]

    recall_, ndcg, precision, f1 = evaluate_one_user(topk, ranking_user, pos_test_items)

    true_labels = np.zeros((len(pred_user, )))
    true_labels[pos_test_items] = 1.

    # try:
    #     # auc.append(roc_auc_score(true_labels, pred_user))
    #     auc = roc_auc_score(true_labels, pred_user)
    # except Exception:
    #     auc = 0.5

    return recall_, ndcg, precision, f1


def evaluate_embedding_parellel_spar(embed_user, item_embedding, train_list, pos_test_items):
    precision, recall_, f1, ndcg, auc = [], [], [], [], None
    # invalide_item = list(set(train_list).intersection(set(pos_test_items)))
    topk = [20, 50, 100, 150, 200]
    # pred_user = embed_user.dot(item_embedding.transpose())
    pred_user = pdist2(embed_user, item_embedding)

    pred_user = 1. - pred_user
    pred_user = sigmoid(pred_user)
    # try:
    #     pred_user[invalide_item] = 0.
    # except:
    #     pass
    ranking_user = np.argsort(- pred_user)

    recall_, ndcg, precision, f1 = evaluate_one_user(topk, ranking_user, pos_test_items)

    true_labels = np.zeros((len(pred_user, )))
    true_labels[pos_test_items] = 1.

    # try:
    #     # auc.append(roc_auc_score(true_labels, pred_user))
    #     auc = roc_auc_score(true_labels, pred_user)
    # except Exception:
    #     auc = 0.5

    return recall_, ndcg, precision, f1

def retrieval_item_hash(user_hash, item_hash, cluster, cluster_dict):
    top = 500
    top_cluster = 10.
    cluster_dist = pdist2(user_hash, cluster)
    min, median = np.min(cluster_dist), np.median(cluster_dist)
    filter = (min + median) / 2
    index_cluster = list(np.where(cluster_dist <= filter)[0])
    # index_cluster = np.argsort(- cluster_dist)
    # index_cluster = index_cluster[0:top_cluster]
    recall_item = [y for i in index_cluster for y in cluster_dict[i]]
    recall_item_hash = item_hash[recall_item]
    candidate_ = 1. - pdist2(user_hash, recall_item_hash)
    candidate_ = np.argsort(- candidate_)
    candidate = [recall_item[i] for i in candidate_[0:top]]
    return candidate


def evaluate_embedding_cont_spar(embed_user, item_embedding, user_hash, item_hash, cluster, cluster_dict,
                                 train_list, pos_test_items):
    candidate = retrieval_item_hash(user_hash, item_hash, cluster, cluster_dict)
    topk = [20, 50, 100, 150, 200]
    item_embedding = item_embedding[candidate]
    # invalide_item = list(set(train_list).intersection(set(pos_test_items)))
    pred_user = embed_user.dot(item_embedding.transpose())
    pred_user = sigmoid(pred_user)
    # try:
    #     pred_user[invalide_item] = 0.
    # except:
    #     pass
    ranking_user_ = list(np.argsort(- pred_user))
    ranking_user = [candidate[index_] for index_ in ranking_user_]

    recall_, ndcg, precision, f1 = evaluate_one_user(topk, ranking_user, pos_test_items)

    return recall_, ndcg, precision, f1

def evaluate_embedding_parellel_delete(embed_user, item_embedding, training_items, pos_test_items):
    precision, recall_, f1, ndcg, auc = [], [], [], [], None
    topk = [20, 50, 100, 150, 200]
    pred_user = embed_user.dot(item_embedding.transpose()).toarray()[0]
    pred_user = sigmoid(pred_user)
    pred_user[training_items] = 0.
    ranking_user = np.argsort(- pred_user)

    recall_, ndcg, precision, f1 = evaluate_one_user(topk, ranking_user, pos_test_items)

    true_labels = np.zeros((len(pred_user, )))
    true_labels[pos_test_items] = 1.

    try:
        # auc.append(roc_auc_score(true_labels, pred_user))
        auc = roc_auc_score(true_labels, pred_user)
    except Exception:
        auc = 0.5

    return recall_, ndcg, precision, f1, auc


def evaluate_embedding_parellel_faiss(embed_user, item_embedding, user_recall, training_items, pos_test_items):
    precision, recall_, f1, ndcg, auc = [], [], [], [], None
    topk = [10, 30, 50, 70, 90, 100, 120, 150, 200]
    pred_user = embed_user.dot(item_embedding.transpose())
    pred_user = sigmoid(pred_user)
    ranking_user_ = list(np.argsort(- pred_user))
    ranking_user = [user_recall[index_] for index_ in ranking_user_]

    recall_, ndcg, precision, f1 = evaluate_one_user(topk, ranking_user, pos_test_items)

    return recall_, ndcg, precision, f1


def evaluate_embedding_gmnn(ranking_user, pos_test_items):
    topk = [10, 50, 100]
    recall_, ndcg, hit_rate = evaluate_one_user_gmnn(topk, ranking_user, pos_test_items)

    return ndcg, hit_rate


def evaluation(input_data, train_label_dict, test_label_dict,
                                sess, topk, model, dim):
    user_embedding, item_embedding = embedding_generation_batch(input_data, sess, model, dim)

    recall_, ndcg, auc, precision, f1 = evaluate_embedding(user_embedding, item_embedding, train_label_dict, test_label_dict, topk)
    print('testing results \n')
    for k in range(len(topk)):
        print('\n topk={} recall={} ndcg={} auc={} precision={} f1={}'.format(topk[k], recall_[k], ndcg[k], auc, precision[k], f1[k]))


def evaluate_full(sess, test_data, model, dim):
    topN = 100
    topk = [50, 100]
    if mp.cpu_count() > 4:
        cores = 4
    else:
        cores = 1
    pool = mp.Pool(cores)

    item_embs = model.output_item(sess)
    # item_embs = model.output_item2(sess)
    index = faiss.IndexFlatIP(dim)
    index.add(item_embs)

    total_ndcg = []
    total_hitrate = []

    inference_time = []

    while True:
        try:
            hist_item, nbr_mask, i_ids = test_data.next()
        except StopIteration:
            break
        t1 = time.time()
        user_embs = model.output_user(sess, hist_item, nbr_mask)
        t2 = time.time()
        inference_time.append(t2 - t1)
        if len(user_embs.shape) == 2:
            D, recalled = index.search(user_embs, topN)
            target_ids = [[i] for i in i_ids]
            results_ = [
                pool.apply(evaluate_embedding_gmnn, args=(recalled[i], iid_list)) for i, iid_list in enumerate(target_ids)]
            ndcg = [user_result[0] for user_result in results_]
            hitrate = [user_result[1] for user_result in results_]
            total_ndcg.extend(ndcg)
            total_hitrate.extend(hitrate)
        else:
            batch_size, interest_num = user_embs.shape[0], user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, recalled = index.search(user_embs, topN)
            recall_ = np.reshape(recalled, [batch_size, interest_num * topN])
            D_recall_ = np.reshape(D, [batch_size, interest_num * topN])
            recalled = []
            for i in range(batch_size):
                item_list = list(zip(np.reshape(recall_[i], [-1]), np.reshape(D_recall_[i], [-1])))
                item_list.sort(key=lambda x: x[1], reverse=True)
                temp_recal = []
                for xx in range(len(item_list)):
                    curr_item = item_list[xx][0]
                    if curr_item not in temp_recal:
                        temp_recal.append(curr_item)
                        if len(temp_recal) == topN:
                            break
                recalled.append(temp_recal)

            target_ids = [[i] for i in i_ids]
            results_ = [
                pool.apply(evaluate_embedding_gmnn, args=(recalled[i], iid_list)) for i, iid_list in
                enumerate(target_ids)]
            ndcg = [user_result[0] for user_result in results_]
            hitrate = [user_result[1] for user_result in results_]
            total_ndcg.extend(ndcg)
            total_hitrate.extend(hitrate)

    pool.close()
    total_ndcg = np.array(total_ndcg)
    total_hitrate = np.array(total_hitrate)
    print('total inference time is %.5f' % sum(inference_time))

    ndcg = np.mean(total_ndcg, axis=0)
    hitrate = np.mean(total_hitrate, axis=0)
    return {'ndcg': ndcg, 'hitrate': hitrate}


def evaluate_fulllarge(sess, data_loader, model, dim, mode='valid'):
    topN = 100
    topk = [50, 100]
    if mode == 'train':
        flag = 0
    elif mode == 'valid':
        flag = 1
    else:
        flag = 2

    if mp.cpu_count() > 4:
        cores = 4
    else:
        cores = 1
    pool = mp.Pool(cores)

    item_embs = model.output_item(sess)
    index = faiss.IndexFlatIP(dim)
    index.add(item_embs)

    total_ndcg = []
    total_hitrate = []

    inference_time = []
    round = 0

    while True:
        try:
            hist_item, nbr_mask, i_ids = data_loader.next(flag)
        except StopIteration:
            break
        t1 = time.time()
        user_embs = model.output_user(sess, hist_item, nbr_mask)
        t2 = time.time()

        inference_time.append(t2 - t1)
        if len(user_embs.shape) == 2:
            D, recalled = index.search(user_embs, topN)
            target_ids = [[i] for i in i_ids]
            results_ = [
                pool.apply(evaluate_embedding_gmnn, args=(recalled[i], iid_list)) for i, iid_list in enumerate(target_ids)]
            ndcg = [user_result[0] for user_result in results_]
            hitrate = [user_result[1] for user_result in results_]
            total_ndcg.extend(ndcg)
            total_hitrate.extend(hitrate)
        else:
            batch_size, interest_num = user_embs.shape[0], user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, recalled = index.search(user_embs, topN)
            recall_ = np.reshape(recalled, [batch_size, interest_num * topN])
            D_recall_ = np.reshape(D, [batch_size, interest_num * topN])
            recalled = []
            for i in range(batch_size):
                item_list = list(zip(np.reshape(recall_[i], [-1]), np.reshape(D_recall_[i], [-1])))
                item_list.sort(key=lambda x: x[1], reverse=True)
                temp_recal = []
                for xx in range(len(item_list)):
                    curr_item = item_list[xx][0]
                    if curr_item not in temp_recal:
                        temp_recal.append(curr_item)
                        if len(temp_recal) == topN:
                            break
                recalled.append(temp_recal)

            target_ids = [[i] for i in i_ids]
            results_ = [
                pool.apply(evaluate_embedding_gmnn, args=(recalled[i], iid_list)) for i, iid_list in
                enumerate(target_ids)]
            ndcg = [user_result[0] for user_result in results_]
            hitrate = [user_result[1] for user_result in results_]
            total_ndcg.extend(ndcg)
            total_hitrate.extend(hitrate)

    pool.close()
    total_ndcg = np.array(total_ndcg)
    total_hitrate = np.array(total_hitrate)

    print('total inference time is %.5f' % sum(inference_time))

    ndcg = np.mean(total_ndcg, axis=0)
    hitrate = np.mean(total_hitrate, axis=0)
    return {'ndcg': ndcg, 'hitrate': hitrate}