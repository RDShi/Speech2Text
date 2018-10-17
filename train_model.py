import sys
import os
import time
import random
import tensorflow as tf
import numpy as np
import argparse
import importlib
import threading
import json
import datetime
import logging
import utils


def get_config(filename):
    with open(filename, 'r') as f:
        f.readline()
        line = f.readline()
        par = line.strip().split(',')
        lr = float(par[0])
        disp_step = int(par[1])
        evaluate_step = int(par[2])

    return lr, disp_step, evaluate_step


def main(args):

    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), args.model_name)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), args.model_name)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir,args.model_name+'.ckpt')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(os.path.join(log_dir, args.model_name+'.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info('pid:'+str(os.getpid()))
    logger.info('Start to build static graph.')

    os.environ['CUDA_VISIBLE_DEVICES']= args.CUDA_VISIBLE_DEVICES

    # tf.reset_default_graph()

    network = importlib.import_module(args.model_def)

    max_checkpoints = 3

    lstm_model_setting={}
    lstm_model_setting['num_units']=128
    lstm_model_setting['dimension_projection']=100
    lstm_model_setting['attn_length']=10
    lstm_model_setting['num_layers']=3

    test_size = 10
    wav_max_len = 1927
    label_max_len = 37
    words_size = 5706
    feat_size = 39

    with open('data/data_list.json','r') as fout:
        data_list = json.load(fout, encoding='utf-8')
    with open('data/vocabulary.json','r') as fout:
        vocabulary = json.load(fout, encoding='utf-8')

    if args.pretrained_model and not args.finetuning:
        try:
            with open(os.path.join(os.path.dirname(args.pretrained_model),'test_speaker.txt'), 'r') as fid:
                test_segment = fid.read().split('\n')
        except:
            test_segment = random.sample(data_list, test_size)
    else:
        test_segment = random.sample(data_list, test_size)

    with open(os.path.join(model_dir, 'test_speaker.txt'), 'w') as fid:
        fid.write('\n'.join(test_segment))

    train_segment = [segment_name  for segment_name in data_list if segment_name not in test_segment]

    # graph
    x = tf.placeholder(tf.float32, [None, wav_max_len, feat_size], name='inputs')
    y = tf.placeholder(dtype=tf.int32, shape=[None, label_max_len], name='target')
    lr_placeholder = tf.placeholder(tf.float32, name='learning_rate')

    with tf.device('/cpu:0'):
        q = tf.FIFOQueue(args.batch_size*3, [tf.float32, tf.int32], shapes=[[wav_max_len, feat_size], [label_max_len]])
        enqueue_op = q.enqueue_many([x, y])
        x_b, y_b = q.dequeue_many(args.batch_size)

    logits = network.inference(x_b, lstm_model_setting, words_size, args.batch_size)

    indices = tf.where(tf.not_equal(tf.cast(y_b, tf.float32), 0.))
    targets = tf.SparseTensor(indices=indices, values=tf.gather_nd(y_b, indices) - 1, dense_shape=tf.cast(tf.shape(y_b), tf.int64))

    sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(x_b, reduction_indices=2), 0.), tf.int32), reduction_indices=1)

    with tf.name_scope('loss'):
        ctc_loss = tf.nn.ctc_loss(targets, logits, sequence_len)
        loss = tf.reduce_mean(ctc_loss)
        tf.summary.scalar('train', loss)

    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, sequence_len, merge_repeated=False)
    # decoded, _ = tf.nn.ctc_greedy_decoder(logits, sequence_len, merge_repeated=False)

    with tf.name_scope('label_error_rate'):
        ler  = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
        tf.summary.scalar('train', ler)

    result = tf.sparse_to_dense(sparse_indices=decoded[0].indices, output_shape=[args.batch_size, wav_max_len], sparse_values = decoded[0].values+1, default_value=0, name='result')

    opt = tf.train.AdamOptimizer(learning_rate=lr_placeholder)
    # opt = tf.train.MomentumOptimizer(learning_rate=lr_placeholder, momentum=0.9, use_nesterov=True)
    global_step = tf.Variable(0, trainable=False)
    # gradients = opt.compute_gradients(loss)
    # capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    # trainer = opt.apply_gradients(capped_gradients, global_step=global_step)
    trainer = opt.minimize(loss, global_step=global_step)

    logger.info('The static graph is completed and ready to start the session.')

    # merge ummaries to write them to file
    merged = tf.summary.merge_all()

    # checkpoint saver and restorer
    if args.only_weight:
        all_vars = tf.trainable_variables()
        excl_vars = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
        to_restore = [item for item in all_vars if tflearn.utils.check_restore_tensor(item, excl_vars)]
    elif args.finetuning:
        all_vars = tf.global_variables()
        excl_vars = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
        to_restore = [item for item in all_vars if tflearn.utils.check_restore_tensor(item, excl_vars)]
    else:
        to_restore = None

    restorer = tf.train.Saver(var_list=to_restore, max_to_keep=max_checkpoints)

    saver = tf.train.Saver(max_to_keep=max_checkpoints)

    coord = tf.train.Coordinator()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    logger.info('The session starts running.')

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.pretrained_model:
            restorer.restore(sess, args.pretrained_model)

        # enqueuing batches procedure
        def enqueue_batches():
            while not coord.should_stop():
                batch_feats, batch_labels  = utils.get_batch(args.batch_size, vocabulary, train_segment)
                sess.run(enqueue_op, feed_dict={x: batch_feats, y: batch_labels})

        # creating and starting parallel threads to fill the queue
        num_threads = 2
        for i in range(num_threads):
            t = threading.Thread(target=enqueue_batches)
            t.setDaemon(True)
            t.start()

        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        start_time = time.time()

        step = sess.run(global_step)

        while step <= args.max_step:

            if args.config_file:
                lr, display_step, evaluate_step = get_config(args.config_file)
            else:
                lr = 0.001
                display_step = 100
                evaluate_step = 1000

            _, step = sess.run([trainer, global_step], feed_dict={lr_placeholder: lr})

            if step % display_step == 0:
                train_loss, train_ler, train_merged = sess.run([loss, ler, merged], feed_dict={lr_placeholder: lr})

                int_time = time.time()
                logger.info('Step: {:09d} --- Loss: {:.7f} Training Label Error Rate: {:.4f} Learning Rate: {} PID: {} Elapsed time: {}'
                            .format(step, train_loss, train_ler, lr, os.getpid(), utils.format_time(int_time - start_time)))
                train_writer.add_summary(train_merged, step)

            if step % evaluate_step == 0:

                logger_info = []

                train_feats, train_labels = utils.get_batch(args.batch_size, vocabulary, train_segment)
                ans = sess.run(result, feed_dict={x_b: train_feats})
                logger_info.append('=================================train result=====================================')
                for i in range(args.batch_size):
                    lab = []
                    labp = []
                    for j in range(wav_max_len):
                        if j == len(train_labels[i]) or sum(train_labels[i][j:]) == 0:
                            break
                        lab.append(vocabulary[train_labels[i][j]-1])
                    logger_info.append('target: {}'.format(' '.join(lab)))
                    for j in range(wav_max_len):
                        if j == len(ans[i]) or sum(ans[i][j:]) == 0:
                            break
                        labp.append(vocabulary[ans[i][j]-1])
                    logger_info.append('result: {}'.format(' '.join(labp)))
                    

                test_feats, test_labels = utils.get_batch(args.batch_size, vocabulary, test_segment)
                ans, test_loss, test_ler = sess.run([result, loss, ler], feed_dict={x_b: test_feats, y_b: test_labels})
                logger_info.append('=================================test result======================================')
                for i in range(args.batch_size):
                    lab = []
                    labp = []
                    for j in range(wav_max_len):
                        if j == len(test_labels[i]) or sum(test_labels[i][j:]) == 0:
                            break
                        lab.append(vocabulary[test_labels[i][j]-1])
                    logger_info.append('target: {}'.format(' '.join(lab)))
                    for j in range(wav_max_len):
                        if j == len(ans[i]) or sum(ans[i][j:]) == 0:
                            break
                        labp.append(vocabulary[ans[i][j]-1])
                    logger_info.append('result: {}'.format(' '.join(labp)))
                logger_info.append('==================================================================================')
                int_time = time.time()
                logger_info.append('Validation --- Validation Label Error Rate: {:.04f}, Elapsed time: {}'.format(test_ler, utils.format_time(int_time - start_time)))
                # save weights to file
                save_path = saver.save(sess, model_path)
                logger_info.append('Variables saved in file: {}'.format(save_path))
                summary = tf.Summary()
                summary.value.add(tag='label_error_rate/val', simple_value=test_ler)
                summary.value.add(tag='loss/val', simple_value=test_loss)
                train_writer.add_summary(summary, step)
                logger_info.append('Logs saved in dir: {}'.format(log_dir))
                logger_info.append('==================================================================================')

                logger.info('\n'+'\n'.join(logger_info))

        end_time = time.time()
        logger.info('Elapsed time: {}'.format(utils.format_time(end_time - start_time)))
        save_path = saver.save(sess, model_path)
        logger.info('Variables saved in file: {}'.format(save_path))
        logger.info('Logs saved in dir: {}'.format(log_dir))

        coord.request_stop()
        coord.join()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, help='pretrained model path.', default=None)
    parser.add_argument('--max_step', type=int, help='Number of steps to run.', default=1000000)
    parser.add_argument('--batch_size', type=int, help='batch size.', default=32)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, help='CUDA VISIBLE DEVICES', default='9')
    parser.add_argument('--config_file', type=str, help='File containing the learning rate schedule', default='./data/config.txt')
    parser.add_argument('--model_name', type=str, help='', default='test')
    parser.add_argument('--logs_base_dir', type=str, help='Directory where to write event logs.', default='~/logs/voice/VCTK_s2t/')
    parser.add_argument('--models_base_dir', type=str, help='Directory where to write trained models and checkpoints.', default='~/models/voice/VCTK_s2t/')
    parser.add_argument('--finetuning', type=bool, help='Whether finetuning.', default=False)
    parser.add_argument('--only_weight', type=bool, help="Whether only load pretrained model's weight.", default=False)
    parser.add_argument('--model_def', type=str, help='Model definition. Points to a module containing the definition of the inference graph.', default='models.lstm')
    parser.add_argument('--data_set', type=str, help="data set position.", default='/data/srd/data/VCTK-Corpus/')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

