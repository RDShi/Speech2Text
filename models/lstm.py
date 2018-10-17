import tensorflow as tf


def lstm_cell(num_proj, attn_length, num_units=128, keep_prob=1.0):
    cell = tf.contrib.rnn.LSTMCell(num_units=num_units, num_proj=num_proj)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.AttentionCellWrapper(cell,attn_length=attn_length) #add att
    return cell

def inference(incoming, lstm_model_setting, words_size, batch_size, keep_prob=1.0, reuse=tf.AUTO_REUSE, finetuning=False):

    with tf.variable_scope('LSTM', reuse=reuse):
        incoming = tf.unstack(incoming, incoming.get_shape().as_list()[1], 1)

        num_proj_list = [lstm_model_setting['dimension_projection']]*(lstm_model_setting['num_layers']-1) + [words_size]
        mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(num_proj, attn_length=lstm_model_setting['attn_length'], 
                                                            num_units=lstm_model_setting['num_units'], keep_prob=keep_prob) 
                                                  for num_proj in num_proj_list])
        initial_state = mlstm_cell.zero_state(batch_size, tf.float32)
        outputs, _ = tf.nn.static_rnn(mlstm_cell, incoming, initial_state= initial_state, dtype=tf.float32)
 
        logits = tf.stack(outputs) # [max_time x batch_size x num_classes]

    return logits


