"""GaussianPolicy."""

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from softlearning.distributions.squash_bijector import SquashBijector
from softlearning.models.feedforward import feedforward_model

from .base_policy import LatentSpacePolicy,DPolicy


SCALE_DIAG_MIN_MAX = (-20, 2)


class DeterministicsPolicy(DPolicy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 eps=0.1,
                 squash=True,
                 preprocessor=None,
                 name=None,
                 *args,
                 **kwargs):
        self._Serializable__initialize(locals())
        self.eps=eps
        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._squash = squash
        self._name = name
        self._preprocessor = preprocessor

        super(DeterministicsPolicy, self).__init__(*args, **kwargs)

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        conditions = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis=-1)
        )(self.condition_inputs)

        if preprocessor is not None:
            conditions = preprocessor(conditions)

        shift = self._shift_and_log_scale_diag_net(
            input_shapes=(conditions.shape[1:], ),
            output_size=output_shape[0],
        )(conditions)

        batch_size = tf.keras.layers.Lambda(
            lambda x: tf.shape(x)[0])(conditions)

        squash_bijector = (SquashBijector()
                           if self._squash
                           else tfp.bijectors.Identity())

        deterministic_actions = tf.keras.layers.Lambda(
            lambda shift: squash_bijector.forward(shift)
        )(shift)

        self.deterministic_actions_model = tf.keras.Model(
            self.condition_inputs, deterministic_actions)

        def raw_actions_fn(inputs):
            shift,eps = inputs
            #actions=shift+tf.random.normal(shape=tf.shape(shift), stddev=eps)
            actions = tf.keras.layers.GaussianNoise(stddev=0.1)(shift)
            #actions = shift + tf.random.normal(shape=tf.shape(shift), stddev=eps)

            return actions

        '''raw_actions = tf.keras.layers.Lambda(
            raw_actions_fn
        )((shift,self.eps))'''

        actions = tf.keras.layers.Lambda(
            lambda deterministic_actions: tf.clip_by_value(deterministic_actions + tf.random.normal(shape=tf.shape(deterministic_actions), stddev=0.1),-1,1)
        )(deterministic_actions)

        '''actions = tf.keras.layers.Lambda(
            raw_actions_fn
        )((deterministic_actions, self.eps))'''

        '''squash_bijector = (SquashBijector()
            if self._squash
            else tfp.bijectors.Identity())'''

        '''actions = tf.keras.layers.Lambda(
            lambda raw_actions: squash_bijector.forward(raw_actions)
        )(raw_actions)'''

        self.actions_model = tf.keras.Model(self.condition_inputs, actions)


        self.actions_input = tf.keras.layers.Input(shape=output_shape)

        self.diagnostics_model = tf.keras.Model(
            self.condition_inputs,
            (shift, actions))

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        raise NotImplementedError

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    @property
    def non_trainable_weights(self):
        """Due to our nested model structure, we need to filter duplicates."""
        return list(set(super(DeterministicsPolicy, self).non_trainable_weights))

    def actions(self, conditions):
        if self._deterministic:
            return self.deterministic_actions_model(conditions)

        return self.actions_model(conditions)

    def actions_np(self, conditions):
        return super(DeterministicsPolicy, self).actions_np(conditions)


    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (shifts_np,
         actions_np) = self.diagnostics_model.predict(conditions)

        return OrderedDict({
            'shifts-mean': np.mean(shifts_np),
            'shifts-std': np.std(shifts_np),

            'actions-mean': np.mean(actions_np),
            'actions-std': np.std(actions_np),
        })


class FeedforwardDeterministicsPolicy(DeterministicsPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='linear',
                 *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._Serializable__initialize(locals())
        super(FeedforwardDeterministicsPolicy, self).__init__(*args, **kwargs)

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        shiftnet = feedforward_model(
            input_shapes=input_shapes,
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)

        return shiftnet

