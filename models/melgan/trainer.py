import tensorflow as tf
from utils.losses import masked_mean_absolute_error

class GANTrainer:
    def __init__(self, generator, discriminator, feature_loss_coeff:float=10., debug:bool=False):
        self.generator = generator
        self.discriminator = discriminator
        self.feature_loss_coeff = feature_loss_coeff
        self.debug = debug
        self.input_signature = [
            tf.TensorSpec(shape=(None, None, self.generator.mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
        ]
        self.adversarial_train_step = self._apply_signature(self._adversarial_train_step, self.input_signature)

    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)

    def _adversarial_train_step(self, mel, wav):
        with tf.GradientTape(persistent=True) as tape:
            pred_wav = self.generator(mel)
            pred_label, pred_features = self.discriminator(pred_wav)
            true_label, true_features = self.discriminator(wav)
            # compute losses
            feature_loss = 0.0
            for i, _ in enumerate(pred_features):
                feature_loss += self._feature_loss(true_feature=true_features[i], pred_feature=pred_features[i])
            feature_loss *= self.feature_loss_coeff
            feature_loss /= float(i+1)
            
            adversarial_loss = self._discriminator_true_label_loss(pred_label)
            generator_loss = adversarial_loss + feature_loss
            
            discriminator_loss_fake = self._discriminator_fake_label_loss(pred_label)
            discriminator_loss_true = self._discriminator_true_label_loss(true_label)
            discriminator_loss = .5 * (discriminator_loss_fake + discriminator_loss_true)
            
        gradients_generator = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        gradients_discriminator = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(gradients_discriminator, self.generator.trainable_variables))
        del tape
        return {'pred_wav': pred_wav,
                'pred_label': pred_label,
                'feature_loss': feature_loss,
                'generator_loss': generator_loss,
                'adversarial_loss': adversarial_loss,
                'discriminator_loss': discriminator_loss,
                'discriminator_loss_fake': discriminator_loss_fake,
                'discriminator_loss_true': discriminator_loss_true,
                }
    
    def _mse_train_step(self, mel, wav):
        with tf.GradientTape() as tape:
            pred_wav = self.generator(mel)
            generator_mse_loss = masked_mean_absolute_error(wav, pred_wav)
        gradients_generator = tape.gradient(generator_mse_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))
        return {'pred_wav': pred_wav,
                'generator_mse_loss': generator_mse_loss}
        
    
    def _discriminator_true_label_loss(self, predicted_labels):
        true_label = tf.ones_like(predicted_labels, dtype=tf.float32)
        loss = masked_mean_absolute_error(true_label, predicted_labels)
        return loss
    
    def _discriminator_fake_label_loss(self, predicted_labels):
        fake_label = tf.zeros_like(predicted_labels, dtype=tf.float32)
        loss = masked_mean_absolute_error(fake_label, predicted_labels)
        return loss
    
    def _feature_loss(self, true_feature, pred_feature):
        loss = masked_mean_absolute_error(true_feature, pred_feature)
        return  loss