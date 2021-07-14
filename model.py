import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
# from tensorflow.nn.rnn_cell import GRUCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.variables import PartitionedVariable

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, share_emb=True, flag="DNN", item_norm=0):
        self.model_flag = flag
        self.reg = False
        self.user_eb = None
        self.batch_size = batch_size
        self.n_size = n_mid
        self.neg_num = 10
        self.lr = 0.001
        self.alpha_para = 0.0
        self.hist_max = seq_len
        self.dim = embedding_dim
        self.share_emb = share_emb
        self.item_norm = item_norm
        with tf.name_scope('Inputs'):
            self.i_ids = tf.placeholder(shape=[None], dtype=tf.int32)
            self.item = tf.placeholder(shape=[None, seq_len], dtype=tf.int32)
            self.nbr_mask = tf.placeholder(shape=[None, seq_len], dtype=tf.float32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.item_input_lookup = tf.get_variable("input_embedding_var", [n_mid, embedding_dim], trainable=True)
            self.item_input_lookup_var = tf.get_variable("input_bias_lookup_table", [n_mid],
                                                       initializer=tf.zeros_initializer(), trainable=False)
            self.position_embedding = tf.get_variable(
                    shape=[1, self.hist_max, embedding_dim],
                    name='position_embedding')
            if self.share_emb:
                self.item_output_lookup = self.item_input_lookup
                self.item_output_lookup_var = self.item_input_lookup_var
            else:
                self.item_output_lookup = tf.get_variable("output_embedding_var", [n_mid, embedding_dim], trainable=True)
                self.item_output_lookup_var = tf.get_variable("output_bias_lookup_table", [n_mid],
                                                             initializer=tf.zeros_initializer(), trainable=False)

        emb = tf.nn.embedding_lookup(self.item_input_lookup,
                                     tf.reshape(self.item, [-1]))
        self.item_emb = tf.reshape(emb, [-1, self.hist_max, self.dim])
        self.mask_length = tf.cast(tf.reduce_sum(self.nbr_mask, -1), dtype=tf.int32)

        self.item_output_emb = self.output_item2()

    def output_item2(self):
        if self.item_norm:
            item_emb = tf.nn.l2_normalize(self.item_output_lookup, dim=-1)
            return item_emb
        else:
            return self.item_output_lookup

    def _xent_loss(self, user):
        emb_dim = self.dim
        loss = tf.nn.sampled_softmax_loss(
            weights=self.output_item2(),
            biases=self.item_output_lookup_var,
            labels=tf.reshape(self.i_ids, [-1, 1]),
            inputs=tf.reshape(user, [-1, emb_dim]),
            num_sampled=self.neg_num * self.batch_size,
            num_classes=self.n_size,
            partition_strategy='mod',
            remove_accidental_hits=True
        )

        self.loss = tf.reduce_mean(loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        return loss

    def _xent_loss_weight(self, user, seq_multi):
        emb_dim = self.dim
        loss = tf.nn.sampled_softmax_loss(
            weights=self.output_item2(),
            # weights=self.item_output_lookup,
            biases=self.item_output_lookup_var,
            labels=tf.reshape(self.i_ids, [-1, 1]),
            inputs=tf.reshape(user, [-1, emb_dim]),
            num_sampled=self.neg_num * self.batch_size,
            num_classes=self.n_size,
            partition_strategy='mod',
            remove_accidental_hits=True
        )

        regs = self.calculate_interest_loss(seq_multi)

        self.loss = tf.reduce_mean(loss)
        self.reg_loss = self.alpha_para * tf.reduce_mean(regs)
        loss = self.loss + self.reg_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return loss

    def train(self, sess, hist_item, nbr_mask, i_ids):
        feed_dict = {
            self.i_ids: i_ids,
            self.item: hist_item,
            self.nbr_mask: nbr_mask
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.item_output_emb)
        # item_embs = sess.run(self.item_output_lookup)
        return item_embs

    def output_user(self, sess, hist_item, nbr_mask):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.item: hist_item,
            self.nbr_mask: nbr_mask
        })
        return user_embs
    
    def save(self, sess, path):
        # if not os.path.exists(path):
        #     os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + '_model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + '_model.ckpt')
        print('model restored from %s' % path)

    def calculate_interest_loss(self, user_interest):
        norm_interests = tf.nn.l2_normalize(user_interest, -1)
        dim0, dim1, dim2 = get_shape(user_interest)

        interests_losses = []
        for i in range(1, (dim1 + 1) // 2):
            roll_interests = array_ops.concat(
                    (norm_interests[:, i:, :], norm_interests[:, 0:i, :]), axis=1)
            # compute pair-wise interests similarity.
            interests_radial_diffs = math_ops.multiply(
                    array_ops.reshape(norm_interests, [dim0*dim1, dim2]),
                    array_ops.reshape(roll_interests, [dim0*dim1, dim2]))
            interests_loss = math_ops.reduce_sum(interests_radial_diffs, axis=-1)
            interests_loss = array_ops.reshape(interests_loss, [dim0, dim1])
            interests_loss = math_ops.reduce_sum(interests_loss, axis=-1)
            interests_losses.append(interests_loss)

        if dim1 % 2 == 0:
            half_dim1 = dim1 // 2
            interests_part1 = norm_interests[:, :half_dim1, :]
            interests_part2 = norm_interests[:, half_dim1:, :]
            interests_radial_diffs = math_ops.multiply(
                    array_ops.reshape(interests_part1, [dim0*half_dim1, dim2]),
                    array_ops.reshape(interests_part2, [dim0*half_dim1, dim2]))
            interests_loss = math_ops.reduce_sum(interests_radial_diffs, axis=-1)
            interests_loss = array_ops.reshape(interests_loss, [dim0, half_dim1])
            interests_loss = math_ops.reduce_sum(interests_loss, axis=-1)
            interests_losses.append(interests_loss)

        # NOTE(reed): the original interests_loss lay in [0, 2], so the
        # combination_size didn't divide 2 to normalize interests_loss into
        # [0, 1]
        self._interests_length = None
        if self._interests_length is not None:
            combination_size = math_ops.cast(
                    self._interests_length * (self._interests_length - 1),
                    dtypes.float32)
        else:
            combination_size = dim1 * (dim1 - 1)
        interests_loss = 0.5 + (
                math_ops.reduce_sum(interests_losses, axis=0) / combination_size)

        return interests_loss

class CapsuleNetwork(tf.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, item_eb, mask):
        with tf.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None, bias_initializer=None)
            else:
                w = tf.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout


class SparseNetwork(object):
    def __init__(self, dim, seq_len, num_topic, category_num, topic_table, position_emb, hidden_units=512):
        super(SparseNetwork, self).__init__()
        self.dim = dim
        self.hist_max = seq_len
        self.num_topic = num_topic
        self.topic_table = topic_table
        self.category_num = category_num
        self.hidden_units = hidden_units
        self.position_embedding = position_emb
        self.inferred_category = None
        self.seq_concept = None
        self.temperature = 0.07

    def topic_select(self, input_seq, nbr_mask):
        seq = tf.reshape(input_seq, [-1, self.hist_max, self.dim])
        seq_emb = self.seq_aggre(seq, nbr_mask)    # [B, D]
        topic_logit = self.topic_table.computate_similarity(seq_emb)    # [B, topic_num]
        top_logits, top_index = tf.nn.top_k(topic_logit, self.category_num)     # [B, C]
        top_logits = tf.sigmoid(top_logits)
        return top_logits, top_index

    def category_infer(self, input_seq, nbr_mask):
        top_logit, top_index = self.topic_select(input_seq, nbr_mask)
        topic_embed = self.topic_table.lookup_table(top_index)  # [B, C, D]
        self.inferred_category = topic_embed * tf.tile(tf.expand_dims(top_logit, axis=2), [1, 1, self.dim])

    def intention_assignment(self, norm_seq):
        # norm_seq size is [B, T, D]
        cores = tf.nn.l2_normalize(self.inferred_category, axis=-1)  # [B, C, D]
        cores_t = tf.reshape(tf.tile(tf.expand_dims(cores, axis=1), [1, self.hist_max, 1, 1]),
                             [-1, self.hist_max, self.category_num, self.dim])     # [B, T, C, D]
        cate_logits = tf.reduce_sum(tf.multiply(cores_t, tf.expand_dims(norm_seq, dim=2)), axis=3)  # [B, T, C]
        interest_dist = tf.nn.softmax(cate_logits / self.temperature, dim=2)
        self.seq_concept = interest_dist
        return interest_dist

    def attention_weight(self, input_seq):
        prob_item = layers.fully_connected(
            input_seq, self.hidden_units, activation_fn=tf.nn.tanh)  # [B,T,D]
        # noinspection PyUnresolvedReferences
        prob_item = layers.fully_connected(
            prob_item, self.category_num, activation_fn=None)  # [B,T,C]
        prob_item = tf.nn.softmax(prob_item / self.temperature, dim=2)  # [B,T, C]
        return prob_item

    def interest_generator(self, item_emb, nbr_mask):
        mask = tf.expand_dims(nbr_mask, axis=2)  # [B, T, 1]
        self.category_infer(item_emb, nbr_mask)
        itm_emb_shift = item_emb + layers.fully_connected(
            item_emb, self.dim, activation_fn=None)  # [B,T,D]
        item_emb_norm = tf.nn.l2_normalize(itm_emb_shift, -1)

        prob_interest = self.intention_assignment(item_emb_norm)   # [batch, T, C]
        prob_interest = tf.multiply(prob_interest, mask)

        item_pos_emb = item_emb + self.position_embedding  # [B,T,D]
        prob_position = self.attention_weight(item_pos_emb)  # [B, T, C]
        prob_position = tf.multiply(prob_position, mask)

        multi_emb = tf.expand_dims(item_emb, 2)     # [B, T, C, D]
        seq = tf.multiply(multi_emb, tf.expand_dims(prob_interest, axis=3))
        seq = tf.multiply(seq, tf.expand_dims(prob_position, axis=3))   # [B, T, C, D]

        seq = tf.multiply(seq, tf.expand_dims(mask, 3))

        seq = tf.reduce_sum(seq, axis=1)    # [B, C, D]

        return seq

    def seq_aggre(self, item_list_emb, nbr_mask):
        num_aggre = 1
        # item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        item_list_add_pos = item_list_emb
        with tf.variable_scope("self_atten_aggre", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w = tf.layers.dense(item_hidden, num_aggre, activation=None)
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, num_aggre, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            item_emb = tf.reshape(item_emb, [-1, self.dim])

            item_emb = tf.nn.l2_normalize(item_emb, dim=-1)
            return item_emb

    def labeled_attention(self, seq_multi, nbr_mask):
        prob_dist = self.seq_concept    # [B, T, C]
        infered_topic = self.inferred_category      # [B, C, D]
        concept_seq = tf.matmul(prob_dist, infered_topic)   # [B, T, D]
        concept_pred = self.sequence_encode_concept(concept_seq, nbr_mask)  #[B, D]

        mu_seq = tf.reduce_mean(seq_multi, axis=1)  # [N,C,D] -> [N,D]
        target_label = tf.concat([mu_seq, concept_pred], axis=1)

        mu = tf.layers.dense(target_label, self.dim, name='maha_cpt2', reuse=tf.AUTO_REUSE)

        wg = tf.matmul(seq_multi, tf.expand_dims(mu, axis=-1))  # (B, C,D)x(B, D,1) -> [B, C, 1]
        wg = tf.nn.softmax(wg, dim=1)

        user_emb = tf.reduce_sum(seq_multi * wg, axis=1)  # [N,C,D] * [N,C, 1] -- > [B, D]
        # user_emb = tf.nn.l2_normalize(user_emb, dim=-1)
        return user_emb

    def sequence_encode_concept(self, item_emb, nbr_mask):

        item_list_emb = tf.reshape(item_emb, [-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + self.position_embedding

        with tf.variable_scope("self_atten_cpt", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, 1, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, 1, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            seq = tf.reshape(item_emb, [-1, self.dim])

        return seq


class PrototypicalDispatcher(object):
    def __init__(self, num_topics, emb_dim, use_cosine=True, temperature=0.07,
                 scope='proto_dispatcher', reuse=None):
        self.num_topics = num_topics
        self.temperature = 1.0
        self.dim = emb_dim
        self.num_topics = num_topics
        self.proto_embs = get_param_var(
            name='proto_embs', shape=[num_topics, emb_dim], reuse=reuse,
            initializer=get_emb_initializer(emb_dim), scope=scope)
        if use_cosine:
            self.temperature = temperature
            self.proto_embs = tf.nn.l2_normalize(self.proto_embs, -1)

    # You need to l2-normalize item_emb by yourself if cosine is enabled!
    def computate_similarity(self, input_seq):
        proto_diss = tf.matmul(input_seq, self.proto_embs, transpose_b=True)     #  [batch_size, topic_num]
        return proto_diss

    def lookup_table(self, top_index):
        topic_embed = tf.nn.embedding_lookup(self.proto_embs, top_index)
        return topic_embed

    def dispatch(self, item_emb, take_log=False,
                 hard=False, compute_grad_as_soft=True):
        assert len(get_shape(item_emb)) == 2  # [B,D]
        y = tf.reduce_sum(tf.multiply(
            tf.expand_dims(item_emb, 1), tf.expand_dims(self.proto_embs, 0)
        ), -1)  # [B,1,D]*[1,H,D] -> [B,H,D] -> [B,H]
        if take_log:
            y = tf.nn.log_softmax(y / self.temperature, 1)  # [B,H]
            return y
        y = tf.nn.softmax(y / self.temperature, 1)  # [B,H]
        if hard:
            y_hard = tf.one_hot(
                tf.argmax(y, -1), self.num_topics, dtype=tf.float32)
            if compute_grad_as_soft:
                y = tf.stop_gradient(y_hard - y) + y
            else:
                y = tf.stop_gradient(y_hard)
        return y  # [B,H]


class Model_SINE(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, topic_num, category_num, alpha,
                 neg_num, cpt_feat, user_norm, item_norm, cate_norm, n_head):
        super(Model_SINE, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="SINE", item_norm=item_norm)
        self.num_topic = topic_num
        self.category_num = category_num
        self.hidden_units = hidden_size
        self.alpha_para = alpha
        self.temperature = 0.07
        # self.temperature = 0.1
        self.user_norm = user_norm
        self.item_norm = item_norm
        self.cate_norm = cate_norm
        self.neg_num = neg_num
        self.num_heads = n_head
        if cpt_feat == 1:
            self.cpt_feat = True
        else:
            self.cpt_feat = False
        with tf.variable_scope('topic_embed', reuse=tf.AUTO_REUSE):
            self.topic_embed = \
                tf.get_variable(
                    shape=[self.num_topic, self.dim],
                    name='topic_embedding')

        self.seq_multi = self.sequence_encode_cpt(self.item_emb, self.nbr_mask)
        self.user_eb = self.labeled_attention(self.seq_multi)
        self._xent_loss_weight(self.user_eb, self.seq_multi)

    def sequence_encode_concept(self, item_emb, nbr_mask):

        item_list_emb = tf.reshape(item_emb, [-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten_cpt", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, self.num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            seq = tf.reshape(item_emb, [-1, self.num_heads, self.dim])
            if self.num_heads != 1:
                mu = tf.reduce_mean(seq, axis=1)
                mu = tf.layers.dense(mu, self.dim, name='maha_cpt')
                wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))
                wg = tf.nn.softmax(wg, dim=1)
                seq = tf.reduce_mean(seq * wg, axis=1)
            else:
                seq = tf.reshape(seq, [-1, self.dim])
        return seq

    def labeled_attention(self, seq):
        # item_emb = tf.reshape(self.cate_dist, [-1, self.hist_max, self.category_num])
        item_emb = tf.transpose(self.cate_dist, [0, 2, 1])
        item_emb = tf.matmul(item_emb, self.batch_tpt_emb)

        if self.cpt_feat:
            item_emb = item_emb + tf.reshape(self.item_emb, [-1, self.hist_max, self.dim])
        target_item = self.sequence_encode_concept(item_emb, self.nbr_mask)#[N,  D]

        mu_seq = tf.reduce_mean(seq, axis=1)  # [N,H,D] -> [N,D]
        target_label = tf.concat([mu_seq, target_item], axis=1)

        mu = tf.layers.dense(target_label, self.dim, name='maha_cpt2', reuse=tf.AUTO_REUSE)

        wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1)
        wg = tf.nn.softmax(wg, dim=1)

        user_emb = tf.reduce_sum(seq * wg, axis=1)  # [N,H,D]->[N,D]
        if self.user_norm:
            user_emb = tf.nn.l2_normalize(user_emb, dim=-1)
        return user_emb

    def seq_aggre(self, item_list_emb, nbr_mask):
        num_aggre = 1
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        with tf.variable_scope("self_atten_aggre", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w = tf.layers.dense(item_hidden, num_aggre, activation=None)
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, num_aggre, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            item_emb = tf.reshape(item_emb, [-1, self.dim])

            return item_emb

    def topic_select(self, input_seq):
        seq = tf.reshape(input_seq, [-1, self.hist_max, self.dim])
        seq_emb = self.seq_aggre(seq, self.nbr_mask)
        if self.cate_norm:
            seq_emb = tf.nn.l2_normalize(seq_emb, dim=-1)
            topic_emb = tf.nn.l2_normalize(self.topic_embed, dim=-1)
            topic_logit = tf.matmul(seq_emb, topic_emb, transpose_b=True)
        else:
            topic_logit = tf.matmul(seq_emb, self.topic_embed, transpose_b=True)#[batch_size, topic_num]
        top_logits, top_index = tf.nn.top_k(topic_logit, self.category_num)#two [batch_size, categorty_num] tensors
        top_logits = tf.sigmoid(top_logits)
        return top_logits, top_index

    def seq_cate_dist(self, input_seq):
        #     input_seq [-1, dim]
        top_logit, top_index = self.topic_select(input_seq)
        topic_embed = tf.nn.embedding_lookup(self.topic_embed, top_index)
        self.batch_tpt_emb = tf.nn.embedding_lookup(self.topic_embed, top_index)#[-1, cate_num, dim]
        self.batch_tpt_emb = self.batch_tpt_emb * tf.tile(tf.expand_dims(top_logit, axis=2), [1, 1, self.dim])
        norm_seq = tf.expand_dims(tf.nn.l2_normalize(input_seq, dim=1), axis=-1)#[-1, dim, 1]
        cores = tf.nn.l2_normalize(topic_embed, dim=-1) #[-1, cate_num, dim]
        cores_t = tf.reshape(tf.tile(tf.expand_dims(cores, axis=1), [1, self.hist_max, 1, 1]), [-1, self.category_num, self.dim])
        cate_logits = tf.reshape(tf.matmul(cores_t, norm_seq), [-1, self.category_num]) / self.temperature #[-1, cate_num]
        cate_dist = tf.nn.softmax(cate_logits, dim=-1)
        return cate_dist

    def sequence_encode_cpt(self, items, nbr_mask):
        item_emb_input = tf.reshape(items, [-1, self.dim])
        # self.cate_dist = tf.reshape(self.seq_cate_dist(self.item_emb), [-1, self.category_num, self.hist_max])
        self.cate_dist = tf.transpose(tf.reshape(self.seq_cate_dist(item_emb_input), [-1, self.hist_max, self.category_num]), [0, 2, 1])
        item_list_emb = tf.reshape(item_emb_input, [-1, self.hist_max, self.dim])
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh, name='fc1')
            item_att_w = tf.layers.dense(item_hidden, self.num_heads * self.category_num, activation=None, name='fc2')

            item_att_w = tf.transpose(item_att_w, [0, 2, 1]) #[batch_size, category_num*num_head, hist_max]

            item_att_w = tf.reshape(item_att_w, [-1, self.category_num, self.num_heads, self.hist_max]) #[batch_size, category_num, num_head, hist_max]

            category_mask_tile = tf.tile(tf.expand_dims(self.cate_dist, axis=2), [1, 1, self.num_heads, 1]) #[batch_size, category_num, num_head, hist_max]
            # paddings = tf.ones_like(category_mask_tile) * (-2 ** 32 + 1)
            seq_att_w = tf.reshape(tf.multiply(item_att_w, category_mask_tile), [-1, self.category_num * self.num_heads, self.hist_max])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.category_num * self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            seq_att_w = tf.where(tf.equal(atten_mask, 0), paddings, seq_att_w)
            seq_att_w = tf.reshape(seq_att_w, [-1, self.category_num, self.num_heads, self.hist_max])

            seq_att_w = tf.nn.softmax(seq_att_w)

            # here use item_list_emb or item_list_add_pos, that is a question
            item_emb = tf.matmul(seq_att_w, tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, self.category_num, 1, 1])) #[batch_size, category_num, num_head, dim]

            category_embedding_mat = tf.reshape(item_emb, [-1, self.num_heads, self.dim]) #[batch_size, category_num, dim]
            if self.num_heads != 1:
                mu = tf.reduce_mean(category_embedding_mat, axis=1)  # [N,H,D]->[N,D]
                mu = tf.layers.dense(mu, self.dim, name='maha')
                wg = tf.matmul(category_embedding_mat, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1) = [N,H,1]
                wg = tf.nn.softmax(wg, dim=1)  # [N,H,1]

                # seq = tf.reduce_mean(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
                seq = tf.reduce_sum(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
            else:
                seq = category_embedding_mat
            self.category_embedding_mat = seq
            seq = tf.reshape(seq, [-1, self.category_num, self.dim])

        return seq


def get_param_var(name, shape, partitioner=None, initializer=None,
                  reuse=None, scope='param'):
    with tf.variable_scope(scope, partitioner=partitioner, reuse=reuse):
        # noinspection PyUnresolvedReferences
        var = tf.get_variable(
            name, shape=shape, dtype=tf.float32,
            initializer=(initializer or layers.xavier_initializer()),
            collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                         ops.GraphKeys.MODEL_VARIABLES])
    return var


def get_emb_initializer(emb_sz):
    # initialize emb this way so that the norm is around one
    return tf.random_normal_initializer(mean=0.0, stddev=float(emb_sz ** -0.5))