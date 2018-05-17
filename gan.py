import mido
from mido import MidiFile
from mido import MidiTrack
from mido import MetaMessage
import io
import tensorflow as tf
import tensorflow.contrib.gan as tfgan
import tensorflow.contrib.framework as framework
import os
import pickle
from tensorflow.python.training import monitored_session
import time
from os.path import isfile, join

layers = tf.contrib.layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

from six.moves import xrange
slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

NUMBER_OF_NOTES = 128
NUM_BYTES = 3
MAX_VAL = 127

flags.DEFINE_float(
    'weight_factor', 50000000.0,
    'How much to weight the adversarial loss relative to note loss.')

log_folder = 'log_n_'+ str(NUMBER_OF_NOTES)+'_'+str(int(round(time.time() * 1000)))
#log_folder = 'log_n_128_1526561687829'
print('log folder:' + log_folder)


def store_output_and_check_loss(sess,gan_loss, generator_tensor, prefix ='none', play=True, num_of_samples=5):
        notes = []
        for _ in range(num_of_samples):
            ret = sess.run(generator_tensor)
            notes = notes + list(ret[0])
            #print(str([ int(v*255) for v in images_np]))
        f_notes=[min([127,round(n*MAX_VAL)]) for n in notes ]
        print(f_notes)
        store_array(f_notes,prefix = prefix, play=play)

def store_array(notes,prefix = 'none',play = True):
    mid = MidiFile()
    track = MidiTrack()
    for i in range(0, len(notes), 3):
        message = mido.Message('note_on', note=int(notes[i]), velocity=int(notes[i+1]), time=int(notes[i+2]))
        track.append(message)

    end = MetaMessage('end_of_track')
    track.append(end)

    mid.tracks.append(track)
    if play:
        with mido.open_output(None, autoreset=True) as port:
            for m in mid.play():
                port.send(m)
    mid.save('outputs/'+str(prefix) + str(int(round(time.time() * 1000)))+'.mid')

real = []
if os.path.exists('real.pickle'):
    with open('real.pickle', 'rb') as handle:
        real = pickle.load(handle)
else:
    meta = []
    temp = []
    mypath = 'MIDI'
    for f in ['mond_2.mid']:    #listdir(mypath)
        print('scaning: ' + str(f))
        with io.open(join(mypath, f), 'rb') as file:
            count = 0
            temp = []
            track = MidiFile(file=file)
            for track_i in [track.tracks[1]]:
                mi = len(track_i)-NUMBER_OF_NOTES
                for ix in range(0,mi,int(NUMBER_OF_NOTES)):
                    fragment = []
                    for message in track_i[ix:ix+NUMBER_OF_NOTES+1]:
                        if message.is_meta:
                            meta.append(message)
                            continue
                        if len(message.bytes()) == 3:
                            #print(str(message) + ' : '  + str(message.bytes()) + ' : ' + str(message.time) )
                            fragment+=message.bytes()[1:]
                            fragment+=[message.time]
                            if len(fragment) == (NUMBER_OF_NOTES*NUM_BYTES):
                                temp.append(fragment)
                                break

            for i in temp:
                if len(i) == NUMBER_OF_NOTES*NUM_BYTES:
                    #print(max(i))
                    real.append(i)

    with open('real'+str(NUMBER_OF_NOTES)  +'.pickle', 'wb') as handle:
        pickle.dump(real, handle, protocol=pickle.HIGHEST_PROTOCOL)


#store_array(real[5])

batch_size = 32
noise_dims = 64
noise = tf.random_normal([1,32],dtype=tf.float32)

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.025)


def generator_fn(noise, weight_decay=2.5e-5, is_training=True):
    with framework.arg_scope(
            [],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)), \
         framework.arg_scope([layers.batch_norm], is_training=is_training,
                             zero_debias_moving_mean=True):
        net = layers.fully_connected(noise, 128,normalizer_fn=None)
        net = layers.dropout(net, keep_prob=0.9)
        net = layers.fully_connected(net, 256,normalizer_fn=None)
        #net = layers.dropout(net, keep_prob=0.875)
        net = layers.fully_connected(net, NUMBER_OF_NOTES * NUM_BYTES,normalizer_fn=None)
        # net = layers.unit_norm(net,1)
        return net



def discriminator_fn(fragment, unused_conditioning, weight_decay=2.5e-7,
                     is_training=True):
    with framework.arg_scope(
            [],
            activation_fn=leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(fragment, 64)
        net = layers.dropout(net, keep_prob=0.75)
        net = layers.fully_connected(net, 32)
        net = layers.fully_connected(net, 16, normalizer_fn=layers.batch_norm,activation_fn=tf.tanh)
        return layers.linear(net, 1, normalizer_fn=None,activation_fn=tf.tanh)

real_data_normed =  tf.divide(tf.convert_to_tensor(real, dtype=tf.float32), tf.constant(MAX_VAL, dtype=tf.float32))
chunk_queue = tf.train.slice_input_producer([real_data_normed])


# Build the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=generator_fn,  # you define
    discriminator_fn=discriminator_fn,  # you define
    real_data=chunk_queue,
    generator_inputs=noise)

gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    gradient_penalty_weight=1.0)

l1_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

gan_loss = tfgan.losses.combine_adversarial_loss(gan_loss, gan_model, l1_loss, weight_factor=FLAGS.weight_factor)

train_ops = tfgan.gan_train_ops(gan_model,gan_loss,generator_optimizer=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.85, beta2=0.999, epsilon=1e-5),discriminator_optimizer=tf.train.AdamOptimizer(learning_rate=0.000001, beta1=0.85, beta2=0.999, epsilon=1e-5))
#train_ops.global_step_inc_op = tf.train.get_global_step().assign_add(1)


#store_output_and_check_loss(gan_loss, gan_model.generated_data, gan_model.real_data, num_of_samples=3, prefix='gen',logdir=log_folder)

global_step_tensor = tf.Variable(1, trainable=False, name='global_step')
global_step = tf.train.get_or_create_global_step()
train_step_fn = tfgan.get_sequential_train_steps( train_steps=tf.contrib.gan.GANTrainSteps(10, 10))
with monitored_session.MonitoredTrainingSession(checkpoint_dir=log_folder) as session:
    loss = None
    for y in xrange(1,20):
        for x in xrange(0,500):
            cur_loss, _ = train_step_fn(session, train_ops, global_step, train_step_kwargs={})

            gen_loss_np = session.run(gan_loss.generator_loss)
            dis_loss_np = session.run(gan_loss.discriminator_loss)

            if gen_loss_np < 170:
                store_output_and_check_loss(session, gan_loss, gan_model.generated_data,prefix='final_l_'+str(round(gen_loss_np))+ '_' + str(NUMBER_OF_NOTES) + '_gen_', play=False,num_of_samples=30)
            print('iteration:'+ str(y*x))
            print('Generator loss: %f' % gen_loss_np)
            print('Discriminator loss: %f' % dis_loss_np)






"""
for _ in range(2):
    store_output_and_check_loss(gan_loss, gan_model.generated_data, gan_model.real_data, num_of_samples=30,
                                prefix='gen', logdir=log_folder)
    start = time.time()
    loss = tfgan.gan_train(train_ops,
                           hooks=[tf.train.StopAtStepHook(num_steps=5000)],
                           logdir=log_folder)
    end = time.time()
    print("!!!!!!! training_done and it took:" + str(end - start))
    store_output_and_check_loss(gan_loss, gan_model.generated_data,gan_model.real_data ,num_of_samples=30, prefix='gen', logdir=log_folder)
"""




