"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the python script to train PredAAE

"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{5}"
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
sys.path.append(".")
import tools as t

os.environ["CUDA_VISIBLE_DEVICES"] = f"{0}"
# from tools import Data_preprocessing as t

# *************values setting***********
root_path = "/home/dg321/gitTest/PRI/irp/Ventilation/direk"
ncoeffs = 56
seed = 42
ntimes = 32  # consecutive times for the AAE
step = 1  # step between times
BATCH_SIZE = 64

epochs = 6000 # 6000
latent_dim = 50 #200

learning_rate_ae = 0.0001
learning_rate_d = 0.0005
# *************values setting***********

all_values = np.load(root_path + '/data/all_values_1124.npy')
ncoeffs = all_values.shape[1]
print(ncoeffs)
ntimes = 32
BATCH_SIZE = 32
step = 1
data_ct = t.concat_timesteps(all_values[10000:16000], ntimes, step)
train_ct, test_ct = t.train_test_split(data_ct, testFrac=0.0)
# create dataset
train_dataset, X_train_4d = t.create_dataset(train_ct, ncoeffs, ntimes, BATCH_SIZE)



# Build PredAAE network
encoder = t.make_encoder_aae1(ntimes, ncoeffs, latent_dim)
decoder = t.make_decoder_aae1(ntimes, ncoeffs, latent_dim)
discriminator = t.make_discriminator_aae(latent_dim)

autoencoder = keras.models.Sequential([encoder, decoder])
enc_disc = keras.models.Sequential([encoder, discriminator])


reconstruction_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
generator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
discriminator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)

r_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_ae)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_ae)

d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_d)

# Notice the use of `tf.function` for speeding up calculation
# This annotation causes the function to be "compiled".
@tf.function
def train_step(batch):
    # Autoencoder update
    with tf.GradientTape() as ae_tape:
        encoder_output = encoder(batch, training=True)
        decoder_output = decoder(encoder_output, training=True)
        reconstruction_loss = t.compute_reconstruction_loss(
            batch, decoder_output)

    r_gradients = ae_tape.gradient(
        reconstruction_loss,
        encoder.trainable_variables +
        decoder.trainable_variables)
    r_optimizer.apply_gradients(
        zip(r_gradients, encoder.trainable_variables + decoder.trainable_variables))

    # Discriminator update
    with tf.GradientTape() as d_tape:
        z = encoder(batch, training=True)
        true_z = tf.random.normal(shape=(z.shape))
        fake_output = discriminator(z, training=True)
        true_output = discriminator(true_z, training=True)
        discriminator_loss = t.compute_discriminator_loss(
            fake_output, true_output)
    d_gradients = d_tape.gradient(
        discriminator_loss,
        discriminator.trainable_variables)
    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables))

    # Generator update
    with tf.GradientTape() as g_tape:
        z = encoder(batch, training=True)
        fake_output = discriminator(z, training=True)
        generator_loss = t.compute_generator_loss(fake_output)
    g_gradients = g_tape.gradient(generator_loss, encoder.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, encoder.trainable_variables))

    reconstruction_mean_loss(reconstruction_loss)
    generator_mean_loss(generator_loss)
    discriminator_mean_loss(discriminator_loss)


def train(dataset, epochs):
    hist = []
    reconstruction = []
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        for batch in dataset:
            train_step(batch)

        hist.append([generator_mean_loss.result().numpy(),
                    discriminator_mean_loss.result().numpy()])
        reconstruction.append(reconstruction_mean_loss.result().numpy())

        generator_mean_loss.reset_states()
        discriminator_mean_loss.reset_states()
        reconstruction_mean_loss.reset_states()

        print("encoder loss: ", hist[-1][0]," - ", "discriminator loss: ", hist[-1][1])
        print("autoencoder loss: ", reconstruction[-1])

    return hist, reconstruction


hist, reconstruction = train(train_dataset, epochs=epochs)
print(reconstruction[-1])

# save loss_enc and loss_dis plot
fig, ax = plt.subplots(1,1, figsize=[20,10])
ax.plot(hist)
ax.legend(['loss_enc', 'loss_disc'])
#ax.set_yscale('log')
ax.grid()
fig.savefig(root_path + '/data/models/1124_aae_encdis_loss_n{}_e{}_s{}_l{}.png'.format(ntimes, epochs, step, latent_dim))
plt.close(fig)    # close the figure window

# save loss_reconstruction plot
fig, ax = plt.subplots(1,1, figsize=[16,8])
ax.plot(reconstruction)
ax.legend(['loss_reconstruction'])
ax.set_yscale('log')
ax.grid()
fig.savefig(root_path + '/data/models/1124_aae_recon_loss_n{}_e{}_s{}_l{}.png'.format(ntimes, epochs, step, latent_dim))
plt.close(fig)    # close the figure window

# save trained model
autoencoder.save(root_path +
                 '/data/models/1124_aae_ae_n{}_e{}_s{}_l{}.h5'.format(ntimes, epochs, step, latent_dim))
enc_disc.save(root_path +
              '/data/models/1124_aae_enc_disc_n{}_e{}_s{}_l{}.h5'.format(ntimes, epochs, step, latent_dim))

# autoencoder = load_model(root_path + 'data/models/1124_aae_ae_n{}_e{}_s{}_l{}.h5'.format(ntimes, epochs, step, latent_dim), compile=False)

# encoder, decoder = load_model(root_path + 'data/models/1124_aae_enc_disc_n{}_e{}_s{}_l{}.h5'.format(ntimes, epochs, step, latent_dim)).layers

# *************Predicting***********
scaler = 1
num_sample = 10

nth_sensor = 5

def predict_coding(initial_pred, real_coding, i):
    loss = []
    for epoch in range(20):
        decoder_output = autoencoder.predict(X_train_4d[i*scaler].reshape((1, ntimes, train_ct.shape[2], 1)))
        loss.append(mse_loss(real_coding, decoder_output[:,:(ntimes - 1),:,:]).numpy())
        initial_pred[:,(ntimes - 1):,:,:] = decoder_output[:,(ntimes - 1):,:,:]

    return decoder_output,loss

mse = tf.keras.losses.MeanSquaredError()
def mse_loss(inp, outp):   
    inp = tf.reshape(inp, [-1, codings_size])
    outp = tf.reshape(outp, [-1, codings_size])
    return mse(inp, outp)

X_test_for_conv = X_train_4d
num_coeffs = 56
input_timestamps = 32
prediction_num = 450
codings_size = all_values.shape[1]
# tqdm
n_pred = prediction_num


# Start from time level 0

n = 0

real_value = X_test_for_conv[n].reshape(1,-1)

# Extract value of 0-(m-2) time levels as real value

real_value = real_value[:,:num_coeffs*(input_timestamps - 1)]

real_value = real_value.reshape((1, input_timestamps-1, X_test_for_conv.shape[2], 1))

# Set value of time level m-1 as same as that of time level m-2

initial_pred = np.concatenate((real_value, real_value[:,-1:,:,:]), axis=1)



# Predict a point forward in time (time level m-1)
i = 0
prediction_values,loss = predict_coding(initial_pred, real_value, i)

# Update real value and initial guess

X_predict = list(prediction_values.reshape(-1,num_coeffs))

# prediction of time level m-1

gen_predict = prediction_values[:,(input_timestamps - 1):,:,:]

# Add the predicted value to the real value (time levels 1-(m-1))

real_value = np.concatenate((real_value[:,1:,:,:], gen_predict), axis=1)

# Set value of time level m as same as that of time level m-1

initial_pred = np.concatenate((real_value, real_value[:,-1:,:,:]), axis=1)

for i in range(prediction_num-1):

    prediction_values,loss = predict_coding(initial_pred, real_value, i)

    # Update real value and initial guess

    gen_predict = prediction_values[:,(input_timestamps - 1):,:,:]

    X_predict.append(gen_predict.flatten())

    real_value = np.concatenate((real_value[:,1:,:,:], gen_predict), axis=1)

    initial_pred = np.concatenate((real_value, real_value[:,-1:,:,:]), axis=1)

X_predict = np.array(X_predict)

print(X_predict.shape)

np.save(root_path + '/predictions/1124_X_predict_model{}_prenum{}_10000_16000.npy'.format(epochs, n_pred), X_predict)
print("finished")