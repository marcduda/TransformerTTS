import tensorflow as tf

from utils.losses import masked_mean_absolute_error, masked_onehot_crossentropy_melgan


# class must be child of keras model for autograph to run.
class GANTrainer(tf.keras.models.Model):
    def __init__(self, generator, discriminator, feature_loss_coeff: float = 10., debug: bool = False, **kwargs):
        super(GANTrainer, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.feature_loss_coeff = feature_loss_coeff
        self.debug = debug
        self.input_signature = [
            tf.TensorSpec(shape=(None, None, self.generator.mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
        ]
        self.adversarial_train_step = self._apply_signature(self._adversarial_train_step, self.input_signature)
        self.mse_train_step = self._apply_signature(self._mse_train_step, self.input_signature)
    
    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)
    
    def _adversarial_train_step(self, mel, wav):
        # wav_mask = tf.math.logical_not(tf.math.equal(wav, self.discriminator.mask_value)) # only needed for mse
        with tf.GradientTape() as dtape:
            discriminator_loss_true = 0.0
            true_label, true_features = self.discriminator.forward(wav)
            for scale, _ in enumerate(true_label):
                discriminator_loss_true += self._discriminator_true_label_loss(true_label[scale])
            discriminator_loss_true /= float(scale + 1)
        gradients_discriminator_true = dtape.gradient(discriminator_loss_true, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(
            zip(gradients_discriminator_true, self.discriminator.trainable_variables))
        del dtape
        
        with tf.GradientTape(persistent=True) as tape:
            pred_wav = self.generator.forward(mel)
            pred_label, pred_features = self.discriminator.forward(pred_wav)
            
            # compute losses
            feature_loss = 0.0
            adversarial_loss_for_generator = 0.0
            discriminator_loss_fake = 0.0
            
            for scale, scale_features in enumerate(pred_features):
                adversarial_loss_for_generator += self._discriminator_true_label_loss(pred_label[scale])
                discriminator_loss_fake += self._discriminator_fake_label_loss(pred_label[scale])
                for feature_number, _ in enumerate(scale_features):
                    feature_loss += self._feature_loss(true_feature=true_features[scale][feature_number],
                                                       pred_feature=pred_features[scale][feature_number])
            feature_loss *= self.feature_loss_coeff
            adversarial_loss_for_generator /= float(scale + 1)
            discriminator_loss_fake /= float(scale + 1)
            feature_loss /= float(scale + 1)
            feature_loss /= float(feature_number + 1)
            
            generator_loss = adversarial_loss_for_generator + feature_loss
        
        gradients_generator = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        gradients_discriminator_true = tape.gradient(discriminator_loss_fake, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(
            zip(gradients_discriminator_true, self.discriminator.trainable_variables))
        del tape
        discriminator_loss = .5 * (discriminator_loss_fake + discriminator_loss_true)
        return {'pred_wav': pred_wav,
                'pred_label': pred_label,
                'loss': {
                    'feature_loss': feature_loss,
                    'generator_loss': generator_loss,
                    'adversarial_loss': adversarial_loss_for_generator,
                    'discriminator_loss': discriminator_loss,
                    'discriminator_loss_fake': discriminator_loss_fake,
                    'discriminator_loss_true': discriminator_loss_true,
                }}
    
    def _mse_train_step(self, mel, wav):
        wav_mask = tf.math.logical_not(tf.math.equal(wav, self.discriminator.mask_value)) # only needed for mse
        with tf.GradientTape() as tape:
            pred_wav = self.generator.forward(mel)
            generator_mse_loss = masked_mean_absolute_error(wav, pred_wav, mask=wav_mask)
        gradients_generator = tape.gradient(generator_mse_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        return {'pred_wav': pred_wav,
                'loss': {
                    'generator_mse_loss': generator_mse_loss}}
    
    def _discriminator_true_label_loss(self, predicted_labels):
        label_shape = tf.shape(predicted_labels[:, :, 0]) # remove one-hot dimension from pred
        true_label = tf.one_hot(tf.ones(label_shape, dtype=tf.int32), 2)
        mask = tf.logical_not(tf.reduce_sum(predicted_labels, axis=-1) == 0)
        mask = tf.cast(mask, dtype=tf.int32)
        loss = masked_onehot_crossentropy_melgan(true_label, predicted_labels, mask=mask, smoothing=0.1)
        return loss
    
    def _discriminator_fake_label_loss(self, predicted_labels):
        label_shape = tf.shape(predicted_labels[:, :, 0]) # remove one-hot dimension from pred
        fake_label = tf.one_hot(tf.zeros(label_shape, dtype=tf.int32), 2)
        mask = tf.logical_not(tf.reduce_sum(predicted_labels, axis=-1) == 0)
        mask = tf.cast(mask, dtype=tf.int32)
        loss = masked_onehot_crossentropy_melgan(targets=fake_label, logits=predicted_labels, smoothing=0.1, mask=mask)
        return loss
    
    def _feature_loss(self, true_feature, pred_feature):
        mask = tf.logical_not(tf.reduce_sum(pred_feature, axis=-1) == 0)
        mask = tf.cast(mask, dtype=tf.int32)
        loss = masked_mean_absolute_error(targets=true_feature, logits=pred_feature, mask=mask)
        return loss
